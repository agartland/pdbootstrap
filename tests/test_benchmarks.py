def stat_mean_single_col(arr):
    return np.array([np.mean(arr)])

import time
import pytest
from pdbootstrap import bootstrap_pd, bootstrap_nb
from tests.test_data import single_col_df, multi_col_df
import time
import pytest
from pdbootstrap import bootstrap_pd
from tests.test_data import single_col_df, multi_col_df

@pytest.fixture(scope="session", autouse=True)
def init_benchmark_file():
    with open("BENCHMARK_RESULTS.md", "w") as f:
        f.write("# Benchmark Results\n\n| Test | Time |\n|---|---|\n")

def write_benchmark_result(name, elapsed, nsamps):
    with open("BENCHMARK_RESULTS.md", "a") as f:
        f.write(f"| {name} | {elapsed:.3f} sec for {nsamps} samples |\n")

NSAMPS = 1000

def test_benchmark_bootstrap_pd():
    df = single_col_df()
    def stat(df):
        return {"a": df["a"].mean()}
    start = time.time()
    res = bootstrap_pd.bootci_pd(df, stat, n_samples=NSAMPS)
    elapsed = time.time() - start
    print(f"Pandas bootstrap_pd: {elapsed:.3f} sec for {NSAMPS} samples")
    write_benchmark_result("Pandas bootstrap_pd (single col)", elapsed, NSAMPS)
    assert elapsed < 10

def test_benchmark_multi_col_pd():
    df = multi_col_df()
    def stat(df):
        return {col: df[col].mean() for col in df.columns}
    start = time.time()
    res = bootstrap_pd.bootci_pd(df, stat, n_samples=NSAMPS)
    elapsed = time.time() - start
    print(f"Pandas multi-col bootstrap_pd: {elapsed:.3f} sec for {NSAMPS} samples")
    write_benchmark_result("Pandas bootstrap_pd (multi col)", elapsed, NSAMPS)
    assert elapsed < 15

