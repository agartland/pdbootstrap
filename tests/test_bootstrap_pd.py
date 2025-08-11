import sys; print("PYTEST sys.path head:", sys.path[:3])
import pytest
from pdbootstrap import bootstrap_pd
from tests.test_data import small_nan_df, small_no_var_df, single_col_df, multi_col_df

def test_nan_handling():
    df = small_nan_df()
    def stat(df):
        return {"a": df["a"].mean(), "b": df["b"].mean()}
    res = bootstrap_pd.bootci_pd(df, stat, n_samples=100)
    assert not res.isnull().values.any()

def test_no_variation():
    df = small_no_var_df()
    def stat(df):
        return {"a": df["a"].mean(), "b": df["b"].mean()}
    res = bootstrap_pd.bootci_pd(df, stat, n_samples=100)
    assert (res.loc["a", "0.025"] == res.loc["a", "0.975"]).all()

def test_single_col():
    df = single_col_df()
    def stat(df):
        return {"a": df["a"].mean()}
    res = bootstrap_pd.bootci_pd(df, stat, n_samples=100)
    assert not res.isnull().values.any()

def test_multi_col():
    df = multi_col_df()
    def stat(df):
        return {col: df[col].mean() for col in df.columns}
    res = bootstrap_pd.bootci_pd(df, stat, n_samples=100)
    assert not res.isnull().values.any()
