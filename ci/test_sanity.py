import pandas as pd
import pytest

KNOWN_STOCKS = {"AARTIIND", "ABCAPITAL", "ABFRL", "ADANIENT", "ADANIGAS"}


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv("data/test.csv")


def test_rolling_avg_10_present_and_finite(test_df):
    assert "rolling_avg_10" in test_df.columns
    assert test_df["rolling_avg_10"].notna().all()


def test_rolling_avg_10_in_sane_range_of_close(test_df):
    for stock, g in test_df.groupby("stock_name"):
        lo, hi = g["low"].min(), g["high"].max()
        tolerance = 0.15 * hi
        assert (g["rolling_avg_10"] >= lo - tolerance).all(), f"{stock}: rolling_avg_10 below sane range"
        assert (g["rolling_avg_10"] <= hi + tolerance).all(), f"{stock}: rolling_avg_10 above sane range"


def test_volume_sum_10_present_and_finite(test_df):
    assert "volume_sum_10" in test_df.columns
    assert test_df["volume_sum_10"].notna().all()


def test_volume_sum_10_non_negative(test_df):
    assert (test_df["volume_sum_10"] >= 0).all()
    assert (test_df["volume_sum_10"] >= test_df["volume"]).all()


def test_stock_name_present(test_df):
    assert "stock_name" in test_df.columns
    assert test_df["stock_name"].notna().all()


def test_stock_name_is_known_value(test_df):
    unknown = set(test_df["stock_name"].unique()) - KNOWN_STOCKS
    assert not unknown, f"Unexpected stock_name values found: {unknown}"
