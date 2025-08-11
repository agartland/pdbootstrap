# pdbootstrap
Python package for bootstrap and permutation testing for pandas DataFrames.

## Installation

```sh
pip install .
```

## Usage

```python
import pandas as pd
from pdbootstrap import bootstrap_pd

df = pd.DataFrame({"a": [1, 2, 3, 4]})
def stat(df):
	return {"a": df["a"].mean()}
res = bootstrap_pd.bootci_pd(df, stat, n_samples=1000)
print(res)
```

## Testing

Run all tests with:

```sh
pytest
```

## License

MIT License

