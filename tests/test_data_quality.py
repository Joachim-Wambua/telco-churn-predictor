import numpy as np
import pandas as pd
import pytest

from src.data.quality import MIN_ROWS_CRITICAL, NULL_PCT_CRITICAL, check_data_quality


def test_quality_gate_passes_on_cleaned_data(cleaned_df):
    result = check_data_quality(cleaned_df)
    assert result["success"] is True, f"Unexpected failures: {result['failures']}"


def test_quality_result_has_required_keys(cleaned_df):
    result = check_data_quality(cleaned_df)
    assert set(result.keys()) == {"success", "failures", "warnings", "statistics"}


def test_quality_catches_too_few_rows():
    # Fewer than MIN_ROWS_CRITICAL rows must produce a critical failure.
    tiny_df = pd.DataFrame({"a": range(MIN_ROWS_CRITICAL - 1), "b": range(MIN_ROWS_CRITICAL - 1)})
    result = check_data_quality(tiny_df)

    assert result["success"] is False
    assert any("Row count" in f for f in result["failures"])


def test_quality_catches_high_null_rate():
    # Column with >NULL_PCT_CRITICAL% nulls must fail even when row count is fine.
    n = 200
    null_count = int(n * (NULL_PCT_CRITICAL / 100) + 10)  # just above the threshold
    col_a = [None] * null_count + list(range(n - null_count))
    bad_df = pd.DataFrame({"a": col_a, "b": range(n)})

    result = check_data_quality(bad_df)

    assert result["success"] is False
    assert any("null" in f.lower() for f in result["failures"])


def test_quality_catches_missing_required_column(cleaned_df):
    result = check_data_quality(cleaned_df, required_columns=["nonexistent_column"])

    assert result["success"] is False
    assert any("missing required columns" in f for f in result["failures"])


def test_quality_statistics_match_dataframe(cleaned_df):
    result = check_data_quality(cleaned_df)
    stats = result["statistics"]

    assert stats["total_rows"] == len(cleaned_df)
    assert stats["total_columns"] == len(cleaned_df.columns)
