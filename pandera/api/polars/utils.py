# pylint: disable=cyclic-import
"""Polars validation engine utilities."""

from typing import Dict, List

import narwhals.stable.v1 as nw

from pandera.config import (
    ValidationDepth,
    get_config_context,
    get_config_global,
)


def get_lazyframe_schema(lf: nw.LazyFrame) -> Dict[str, nw.dtypes.DType]:
    """Get a dict of column names and  dtypes from a polars LazyFrame."""
    return lf.collect_schema()


def get_lazyframe_column_dtypes(lf: nw.LazyFrame) -> List[nw.dtypes.DType]:
    """Get a list of column dtypes from a polars LazyFrame."""
    return lf.collect_schema().dtypes()


def get_lazyframe_column_names(lf: nw.LazyFrame) -> List[str]:
    """Get a list of column names from a polars LazyFrame."""
    return lf.collect_schema().names()


def get_validation_depth(*, is_dataframe: bool) -> ValidationDepth:
    """Get validation depth for a given polars check object."""
    config_global = get_config_global()
    config_ctx = get_config_context(validation_depth_default=None)

    if config_ctx.validation_depth is not None:
        # use context configuration if specified
        return config_ctx.validation_depth

    if config_global.validation_depth is not None:
        # use global configuration if specified
        return config_global.validation_depth

    if config_global.validation_depth is None and not is_dataframe:
        # if global validation depth is not set, use schema only validation
        # when validating LazyFrames
        validation_depth = ValidationDepth.SCHEMA_ONLY
    elif is_dataframe and (
        config_ctx.validation_depth is None
        or config_ctx.validation_depth is None
    ):
        # if context validation depth is not set, use schema and data validation
        # when validating DataFrames
        validation_depth = ValidationDepth.SCHEMA_AND_DATA
    else:
        validation_depth = ValidationDepth.SCHEMA_ONLY

    return validation_depth
