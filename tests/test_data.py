import numpy as np
import pandas as pd


# Single column DataFrame
def single_col_df():
    return multi_col_df()[['a']]


# Multi column DataFrame
def multi_col_df():
    np.random.seed(1)
    return pd.DataFrame(np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"])


# DataFrame with NaNs for edge case tests
def small_nan_df():
    return pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [np.nan, 2.0, 3.0]
    })


# DataFrame with no variation for edge case tests
def small_no_var_df():
    return pd.DataFrame({
        "a": [1.0, 1.0, 1.0],
        "b": [2.0, 2.0, 2.0]
    })
