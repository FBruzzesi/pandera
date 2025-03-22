"""Polars Parsing, Validation, and Error Reporting Backends."""

import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import narwhals.stable.v1 as nw

from pandera.api.base.error_handler import ErrorHandler
from pandera.api.polars.types import CheckResult
from pandera.api.polars.utils import get_lazyframe_column_dtypes
from pandera.backends.base import BaseSchemaBackend, CoreCheckResult
from pandera.constants import CHECK_OUTPUT_KEY
from pandera.errors import (
    FailureCaseMetadata,
    SchemaError,
    SchemaErrorReason,
    SchemaWarning,
)


def is_float_dtype(check_obj: nw.LazyFrame, selector):
    """Check if a column/selector is a float."""
    return all(
        dtype.is_float()
        for dtype in get_lazyframe_column_dtypes(
            check_obj.select(nw.col(selector))
        )
    )


class PolarsSchemaBackend(BaseSchemaBackend):
    """Backend for polars LazyFrame schema."""

    def subsample(
        self,
        check_obj: nw.LazyFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> nw.LazyFrame:
        obj_subsample: List[nw.LazyFrame] = []
        if head is not None:
            obj_subsample.append(check_obj.head(head))
        if tail is not None:
            obj_subsample.append(check_obj.tail(tail))
        if sample is not None:
            obj_subsample.append(
                # mypy is detecting a bug https://github.com/unionai-oss/pandera/issues/1912
                check_obj.sample(  # type:ignore [attr-defined]
                    sample,
                    seed=random_state,  # TODO: Narwhals has sample for DataFrame only?
                )
            )
        return (
            check_obj
            if not obj_subsample
            else nw.concat(obj_subsample).unique()
        )

    def run_check(
        self,
        check_obj: nw.LazyFrame,
        schema,
        check,
        check_index: int,
        *args,
    ) -> CoreCheckResult:
        """Handle check results, raising SchemaError on check failure.

        :param check_obj: data object to be validated.
        :param schema: pandera schema object
        :param check: Check object used to validate pandas object.
        :param check_index: index of check in the schema component check list.
        :param args: arguments to pass into check object.
        :returns: True if check results pass or check.raise_warning=True, otherwise
            False.
        """
        check_result: CheckResult = check(check_obj, *args)

        passed = check_result.check_passed.collect().item()
        failure_cases = None
        message = None

        if not passed:
            if check_result.failure_cases is None:
                # encode scalar False values explicitly
                failure_cases = passed
                message = (
                    f"{schema.__class__.__name__} '{schema.name}' failed "
                    f"{check_index}: {check}"
                )
            else:
                # use check_result
                failure_cases = check_result.failure_cases.collect()
                failure_cases_msg = failure_cases.head().rows(named=True)
                message = (
                    f"{schema.__class__.__name__} '{schema.name}' failed "
                    f"validator number {check_index}: "
                    f"{check} failure case examples: {failure_cases_msg}"
                )

            # raise a warning without exiting if the check is specified to do so
            # but make sure the check passes
            if check.raise_warning:
                warnings.warn(
                    message,
                    SchemaWarning,
                )
                return CoreCheckResult(
                    passed=True,
                    check=check,
                    reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                )
        return CoreCheckResult(
            passed=passed,
            check=check,
            check_index=check_index,
            check_output=check_result.check_output.collect(),
            reason_code=SchemaErrorReason.DATAFRAME_CHECK,
            message=message,
            failure_cases=failure_cases,
        )

    def failure_cases_metadata(
        self,
        schema_name: str,
        schema_errors: List[SchemaError],
    ) -> FailureCaseMetadata:
        """Create failure cases metadata required for SchemaErrors exception."""
        error_counts: Dict[str, int] = defaultdict(int)

        failure_case_collection = []

        for err in schema_errors:

            error_counts[err.reason_code] += 1

            check_identifier = (
                None
                if err.check is None
                else (
                    err.check
                    if isinstance(err.check, str)
                    else (
                        err.check.error
                        if err.check.error is not None
                        else (
                            err.check.name
                            if err.check.name is not None
                            else str(err.check)
                        )
                    )
                )
            )

            if isinstance(err.failure_cases, nw.LazyFrame):
                raise NotImplementedError

            if isinstance(err.failure_cases, nw.DataFrame):
                failure_cases_df = err.failure_cases

                # get row number of the failure cases
                _index_lf = err.check_output.with_row_index("index")
                index = _index_lf.filter(~nw.col(CHECK_OUTPUT_KEY))["index"]
                if len(err.failure_cases.columns) > 1:
                    # for boolean dataframe check results, reduce failure cases
                    # to a struct column
                    failure_cases_df = err.failure_cases.with_columns(
                        failure_case=nw.new_series(
                            name="failure_case",
                            values=err.failure_cases.rows(named=True),
                            # dtype=nw.Struct()
                            backend=nw.Implementation.POLARS,
                        )
                    ).select(
                        nw.col("failure_case").struct.json_encode()
                    )  # TODO: implement json_encode
                else:
                    failure_cases_df = err.failure_cases.rename(
                        {err.failure_cases.columns[0]: "failure_case"}
                    )

                failure_cases_df = failure_cases_df.with_columns(
                    failure_case=nw.col("failure_case").cast(nw.String()),
                    schema_context=nw.lit(err.schema.__class__.__name__),
                    column=nw.lit(err.schema.name).cast(nw.String()),
                    check=nw.lit(check_identifier),
                    check_number=nw.lit(err.check_index, dtype=nw.Int32()),
                    index=index.cast(nw.Int32()),
                )

            else:
                scalar_failure_cases = defaultdict(list)
                scalar_failure_cases["failure_case"].append(err.failure_cases)
                scalar_failure_cases["schema_context"].append(
                    err.schema.__class__.__name__
                )
                scalar_failure_cases["column"].append(err.schema.name)
                scalar_failure_cases["check"].append(check_identifier)
                scalar_failure_cases["check_number"].append(err.check_index)
                scalar_failure_cases["index"].append(None)
                failure_cases_df = nw.from_dict(
                    data=scalar_failure_cases, backend=nw.Implementation.POLARS
                ).with_columns(
                    check_number=nw.col("check_number").cast(nw.Int32()),
                    column=nw.col("column").cast(nw.String()),
                    index=nw.col("index").cast(nw.Int32()),
                )

            failure_case_collection.append(failure_cases_df)

        failure_cases = nw.concat(failure_case_collection)

        error_handler = ErrorHandler()
        error_handler.collect_errors(schema_errors)
        error_dicts = {}

        def defaultdict_to_dict(d):
            if isinstance(d, defaultdict):
                d = {k: defaultdict_to_dict(v) for k, v in d.items()}
            return d

        if error_handler.collected_errors:
            error_dicts = error_handler.summarize(schema_name=schema_name)
            error_dicts = defaultdict_to_dict(error_dicts)

        error_counts = defaultdict(int)  # type: ignore
        for error in error_handler.collected_errors:
            error_counts[error["reason_code"].name] += 1

        return FailureCaseMetadata(
            failure_cases=failure_cases.to_native(),
            message=error_dicts,
            error_counts=error_counts,
        )

    def drop_invalid_rows(
        self,
        check_obj: nw.LazyFrame,
        error_handler: ErrorHandler,
    ) -> nw.LazyFrame:
        """Remove invalid elements in a check obj according to failures in caught by the error handler."""
        errors = error_handler.schema_errors
        check_outputs = (
            nw.concat(
                [
                    err.check_output.select(nw.all().name.suffix(f"_{idx}"))
                    for idx, err in enumerate(errors)
                    if err.check_output is not None
                ],
                how="horizontal",
            )
            .lazy()
            .select(valid_rows=nw.all_horizontal(nw.selectors.boolean()))
        )
        return (
            nw.concat([check_obj, check_outputs], how="horizontal")
            .filter(nw.col("valid_rows"))
            .drop(["valid_rows"])
        )
