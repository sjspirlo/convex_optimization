"""
load_industries.py  --  EBC4286 Convex Optimisation, Assignment 1

Reads the Kenneth French "30 Industry Portfolios" file (the frozen copy
posted on Canvas) and returns monthly industry returns, plus the sample
estimates of the annualised mean vector and covariance matrix.

Usage
-----
    from load_industries import load_industry_returns, estimation_inputs

    R = load_industry_returns("30_Industry_Portfolios_CSV.zip")
    mu, Sigma, names = estimation_inputs(R, start="2015-01", end="2024-12")

The file can be passed as the .zip from Canvas or as the extracted .csv.

Conventions
-----------
* Returns are simple monthly returns in DECIMALS (0.01 = 1%).
* Value-weighted industry portfolios are used unless you ask otherwise.
* mu    = 12 * (sample mean of monthly returns)        (annualised)
* Sigma = 12 * (sample covariance of monthly returns)  (annualised)
  A portfolio w then has expected annual return mu @ w and annual
  volatility sqrt(w @ Sigma @ w).
"""

import io
import zipfile

import numpy as np
import pandas as pd

_TABLES = {
    "value": "Average Value Weighted Returns -- Monthly",
    "equal": "Average Equal Weighted Returns -- Monthly",
}


def _read_text(path):
    """Return the contents of the CSV, whether given as .zip or .csv."""
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(f"Expected one CSV in {path}, found {csv_names}")
            raw = z.read(csv_names[0])
    else:
        with open(path, "rb") as f:
            raw = f.read()
    return raw.decode("latin-1")


def load_industry_returns(path="30_Industry_Portfolios_CSV.zip", weighting="value"):
    """
    Monthly industry returns as a DataFrame (rows: months, columns: industries).

    The French file contains several tables one after another (value-weighted
    monthly, equal-weighted monthly, annual returns, firm counts, ...).
    This function extracts only the requested monthly table.
    """
    if weighting not in _TABLES:
        raise ValueError("weighting must be 'value' or 'equal'")
    lines = _read_text(path).splitlines()

    # Locate the title line of the requested table.
    title = _TABLES[weighting]
    try:
        start = next(i for i, ln in enumerate(lines) if title in ln)
    except StopIteration:
        raise ValueError(f"Table '{title}' not found in {path}")

    # Header line follows the title; data lines follow until the first line
    # that does not start with a 6-digit YYYYMM date.
    header = lines[start + 1]
    block = [header]
    for ln in lines[start + 2:]:
        first = ln.split(",")[0].strip()
        if len(first) == 6 and first.isdigit():
            block.append(ln)
        else:
            break

    df = pd.read_csv(io.StringIO("\n".join(block)), index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df.index = pd.PeriodIndex(df.index.astype(str), freq="M")
    df.index.name = "month"

    # French codes missing observations as -99.99 or -999.
    df = df.astype(float).mask(df <= -99.99)
    return df / 100.0


def estimation_inputs(R, start="2015-01", end="2024-12"):
    """
    Sample estimates over the estimation window [start, end] (inclusive).

    Returns
    -------
    mu    : (n,) numpy array, annualised expected returns
    Sigma : (n, n) numpy array, annualised covariance matrix
    names : list of industry names, in the same order
    """
    window = R.loc[start:end]
    if window.isna().any().any():
        bad = window.columns[window.isna().any()].tolist()
        raise ValueError(f"Missing data in estimation window for: {bad}")
    mu = 12.0 * window.mean().to_numpy()
    Sigma = 12.0 * window.cov().to_numpy()   # ddof = 1
    return mu, Sigma, list(window.columns)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "30_Industry_Portfolios_CSV.zip"
    R = load_industry_returns(path)
    print(f"Loaded {R.shape[1]} industries, {R.index[0]} to {R.index[-1]}")

    mu, Sigma, names = estimation_inputs(R)
    n = len(names)
    w_eq = np.ones(n) / n
    print(f"Estimation window 2015-01 to 2024-12: {len(R.loc['2015-01':'2024-12'])} months")
    print(f"Equally weighted portfolio: expected return {mu @ w_eq:.2%}, "
          f"volatility {np.sqrt(w_eq @ Sigma @ w_eq):.2%}")
