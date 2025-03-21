"""Polars types."""

from typing import NamedTuple, Optional, Union

import narwhals.stable.v1 as nw
import polars as pl


class PolarsData(NamedTuple):
    lazyframe: nw.LazyFrame
    key: Optional[str] = None


class CheckResult(NamedTuple):
    """Check result for user-defined checks."""

    check_output: nw.LazyFrame
    check_passed: nw.LazyFrame
    checked_object: nw.LazyFrame
    failure_cases: nw.LazyFrame


PolarsCheckObjects = Union[pl.LazyFrame, pl.DataFrame]

PolarsDtypeInputTypes = Union[
    str,
    type,
    pl.datatypes.classes.DataTypeClass,
]
