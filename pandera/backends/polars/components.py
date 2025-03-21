"""Validation backend for polars components."""

import warnings
from typing import Any, Callable, Iterable, List, Optional, cast

import narwhals.stable.v1 as nw

from pandera.api.base.error_handler import ErrorHandler
from pandera.api.polars.components import Column
from pandera.api.polars.types import PolarsData
from pandera.api.polars.utils import (
    get_lazyframe_schema,
    get_lazyframe_column_names,
)
from pandera.backends.base import CoreCheckResult
from pandera.backends.polars.base import PolarsSchemaBackend, is_float_dtype
from pandera.config import ValidationDepth, ValidationScope, get_config_context
from pandera.constants import CHECK_OUTPUT_KEY
from pandera.errors import (
    ParserError,
    SchemaDefinitionError,
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)
from pandera.validation_depth import validate_scope, validation_type


class ColumnBackend(PolarsSchemaBackend):
    """Column backend for polars LazyFrames."""

    def preprocess(
        self, check_obj: nw.LazyFrame, inplace: bool = False
    ) -> nw.LazyFrame:
        """Returns a copy of the object if inplace is False."""
        # NOTE: is this even necessary?
        return (
            check_obj
            if inplace
            else nw.from_native(check_obj.to_native().clone())
        )
        # Narwhals' LazyFrame does not have a clone method??? DataFrame's do though

    # pylint: disable=too-many-locals
    def validate(
        self,
        check_obj: nw.LazyFrame,
        schema: Column,
        *,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> nw.LazyFrame:

        if inplace:
            warnings.warn("setting inplace=True will have no effect.")

        if schema.name is None:
            raise SchemaDefinitionError(
                "Column schema must have a name specified."
            )

        error_handler = ErrorHandler(lazy)
        check_obj = self.preprocess(check_obj, inplace)

        if getattr(schema, "drop_invalid_rows", False) and not lazy:
            raise SchemaDefinitionError(
                "When drop_invalid_rows is True, lazy must be set to True."
            )

        core_parsers: List[Callable[..., Any]] = [
            self.coerce_dtype,
            self.set_default,
        ]

        for parser in core_parsers:
            try:
                check_obj = parser(check_obj, schema)
            except SchemaError as exc:
                error_handler.collect_error(
                    validation_type(exc.reason_code),
                    exc.reason_code,
                    exc,
                )
            except SchemaErrors as exc:
                error_handler.collect_errors(exc.schema_errors)

        error_handler = self.run_checks_and_handle_errors(
            error_handler,
            schema,
            check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
        )

        if lazy and error_handler.collected_errors:
            if getattr(schema, "drop_invalid_rows", False):
                check_obj = self.drop_invalid_rows(check_obj, error_handler)
            else:
                raise SchemaErrors(
                    schema=schema,
                    schema_errors=error_handler.schema_errors,
                    data=check_obj,
                )

        return check_obj

    def get_regex_columns(self, schema, check_obj: nw.LazyFrame) -> Iterable:
        return get_lazyframe_schema(check_obj.select(nw.col(schema.selector)))

    def run_checks_and_handle_errors(
        self,
        error_handler: ErrorHandler,
        schema,
        check_obj: nw.LazyFrame,
        **subsample_kwargs,
    ):
        """Run checks on schema"""
        # pylint: disable=too-many-locals
        check_obj_subsample = self.subsample(check_obj, **subsample_kwargs)

        core_checks = [
            self.check_nullable,
            self.check_unique,
            self.check_dtype,
            self.run_checks,
        ]
        args = (check_obj_subsample, schema)
        for core_check in core_checks:
            results = core_check(*args)
            if isinstance(results, CoreCheckResult):
                results = [results]
            results = cast(List[CoreCheckResult], results)
            for result in results:
                if result.passed:
                    continue

                if result.schema_error is not None:
                    error = result.schema_error
                else:
                    assert result.reason_code is not None
                    error = SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=result.message,
                        failure_cases=result.failure_cases,
                        check=result.check,
                        check_index=result.check_index,
                        check_output=result.check_output,
                        reason_code=result.reason_code,
                    )
                    error_handler.collect_error(
                        validation_type(result.reason_code),
                        result.reason_code,
                        error,
                        original_exc=result.original_exc,
                    )

        return error_handler

    def coerce_dtype(
        self,
        check_obj: nw.LazyFrame,
        schema=None,
        # pylint: disable=unused-argument
    ) -> nw.LazyFrame:
        """Coerce type of a pd.Series by type specified in dtype.

        :param check_obj: LazyFrame to coerce
        :returns: coerced LazyFrame
        """
        assert schema is not None, "The `schema` argument must be provided."
        if schema.dtype is None or not schema.coerce:
            return check_obj

        config_ctx = get_config_context(validation_depth_default=None)
        coerce_fn: Callable[[nw.LazyFrame], nw.LazyFrame] = (
            schema.dtype.coerce
            if config_ctx.validation_depth == ValidationDepth.SCHEMA_ONLY
            else schema.dtype.try_coerce
        )

        try:
            return coerce_fn(check_obj)
        except ParserError as exc:
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"Error while coercing '{schema.selector}' to type "
                    f"{schema.dtype}: {exc}"
                ),
                check=f"coerce_dtype('{schema.dtype}')",
                reason_code=SchemaErrorReason.DATATYPE_COERCION,
            ) from exc

    @validate_scope(scope=ValidationScope.DATA)
    def check_nullable(
        self,
        check_obj: nw.LazyFrame,
        schema,
    ) -> List[CoreCheckResult]:
        """Check if a column is nullable.

        This check considers nulls and nan values as effectively equivalent.
        """
        if schema.nullable:
            return [
                CoreCheckResult(
                    passed=True,
                    check="not_nullable",
                    reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
                )
            ]

        expr = ~nw.col(schema.selector).is_null()
        if is_float_dtype(check_obj, schema.selector):
            expr = expr & ~nw.col(schema.selector).is_nan()

        isna = check_obj.select(expr)
        passed = isna.select(nw.all().all()).collect()
        results = []

        # TODO: Can this for loop be optimized?
        for column in get_lazyframe_column_names(isna):
            if passed.select(column).item():
                continue
            failure_cases = (
                nw.concat(
                    items=[
                        check_obj,
                        isna.select(nw.col(column).alias(CHECK_OUTPUT_KEY)),
                    ],
                    how="horizontal",
                )
                .filter(~nw.col(CHECK_OUTPUT_KEY))
                .select(column)
                .collect()
            )
            results.append(
                CoreCheckResult(
                    passed=cast(bool, passed.select(column).item()),
                    check_output=isna.collect().rename(
                        {column: CHECK_OUTPUT_KEY}
                    ),
                    check="not_nullable",
                    reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
                    message=(
                        f"non-nullable column '{schema.selector}' contains "
                        f"null values"
                    ),
                    failure_cases=failure_cases,
                )
            )
        return results

    @validate_scope(scope=ValidationScope.DATA)
    def check_unique(
        self,
        check_obj: nw.LazyFrame,
        schema,
    ) -> List[CoreCheckResult]:
        check_name = "field_uniqueness"
        if not schema.unique:
            return [
                CoreCheckResult(
                    passed=True,
                    check=check_name,
                    reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
                )
            ]

        results = []
        duplicates = (
            check_obj.select(schema.selector)
            .select(nw.all().is_duplicated())
            .collect()
        )
        for column in duplicates.columns:
            if duplicates.select(nw.col(column).any()).item():
                failure_cases = (
                    nw.concat(
                        items=[
                            check_obj,
                            duplicates.select(
                                nw.col(column).alias("_duplicated")
                            ).lazy(),
                        ],
                        how="horizontal",
                    )
                    .filter(nw.col("_duplicated"))
                    .select(column)
                    .collect()
                )
                results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check_name,
                        check_output=duplicates.select(
                            ~nw.col(column).alias(CHECK_OUTPUT_KEY)
                        ),
                        reason_code=SchemaErrorReason.SERIES_CONTAINS_DUPLICATES,
                        message=(
                            f"column '{schema.selector}' "
                            f"not unique:\n{failure_cases}"
                        ),
                        failure_cases=failure_cases,
                    )
                )

        return results

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_dtype(
        self,
        check_obj: nw.LazyFrame,
        schema: Column,
    ) -> List[CoreCheckResult]:

        passed = True
        failure_cases = None
        msg = None

        if schema.dtype is None:
            return [
                CoreCheckResult(
                    passed=passed,
                    check=f"dtype('{schema.dtype}')",
                    reason_code=SchemaErrorReason.WRONG_DATATYPE,
                    message=msg,
                    failure_cases=failure_cases,
                )
            ]

        results = []
        check_obj_subset = check_obj.select(nw.col(schema.selector))
        for column, obj_dtype in get_lazyframe_schema(
            check_obj_subset
        ).items():
            results.append(
                CoreCheckResult(
                    passed=schema.dtype.check(  # TODO
                        obj_dtype,
                        PolarsData(check_obj_subset, schema.selector),
                    ),
                    check=f"dtype('{schema.dtype}')",
                    reason_code=SchemaErrorReason.WRONG_DATATYPE,
                    message=(
                        f"expected column '{column}' to have type "
                        f"{schema.dtype}, got {obj_dtype}"
                    ),
                    failure_cases=str(obj_dtype),
                )
            )
        return results

    # pylint: disable=unused-argument
    @validate_scope(scope=ValidationScope.DATA)
    def run_checks(
        self, check_obj: nw.LazyFrame, schema
    ) -> List[CoreCheckResult]:
        check_results: List[CoreCheckResult] = []
        for check_index, check in enumerate(schema.checks):
            try:
                check_results.append(
                    self.run_check(
                        check_obj,
                        schema,
                        check,
                        check_index,
                        schema.selector,
                    )
                )
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
                msg = f"{err.__class__.__name__}({err_msg})"
                check_results.append(
                    CoreCheckResult(
                        passed=False,
                        check=check,
                        check_index=check_index,
                        reason_code=SchemaErrorReason.CHECK_ERROR,
                        message=msg,
                        failure_cases=msg,
                        original_exc=err,
                    )
                )
        return check_results

    def set_default(self, check_obj: nw.LazyFrame, schema) -> nw.LazyFrame:
        """Set default values for columns with missing values."""
        if hasattr(schema, "default") and schema.default is None:
            return check_obj

        if isinstance(schema.default, nw.Expr):
            default_value = schema.default
        else:
            default_value = nw.lit(schema.default, dtype=schema.dtype.type)
        expr = nw.col(schema.selector)
        if is_float_dtype(check_obj, schema.selector):
            # TODO: Implement fill_nan in narwhals
            expr = (
                nw.when(~expr.is_nan())
                .then(expr)
                .otherwise(nw.lit(default_value))
            )
        else:
            expr = expr.fill_null(default_value)

        return check_obj.with_columns(expr)
