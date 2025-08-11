import pytest
from pdbootstrap import bootstrap_pd
from tests.test_data import small_nan_df, small_no_var_df, single_col_df, multi_col_df

def test_perm_nan():
    df = small_nan_df()
    def stat(df):
        return float(df["a"].mean())
    pval = bootstrap_pd.permtest_pd(df, stat, perm_cols=["a"], n_samples=100)
    assert isinstance(pval, float)

def test_perm_no_var():
    df = small_no_var_df()
    def stat(df):
        return float(df["a"].mean())
    pval = bootstrap_pd.permtest_pd(df, stat, perm_cols=["a"], n_samples=100)
    assert isinstance(pval, float)

def test_perm_single_col():
    df = single_col_df()
    def stat(df):
        return float(df["a"].mean())
    pval = bootstrap_pd.permtest_pd(df, stat, perm_cols=["a"], n_samples=100)
    assert isinstance(pval, float)

def test_perm_multi_col():
    df = multi_col_df()
    def stat(df):
        return float(df["a"].mean())
    pval = bootstrap_pd.permtest_pd(df, stat, perm_cols=["a"], n_samples=100)
    assert isinstance(pval, float)
