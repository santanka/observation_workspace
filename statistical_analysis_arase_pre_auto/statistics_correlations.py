"""Descriptive correlations of individual finite range pairs in displayed coordinates."""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def pair_correlations(x, y, log_x=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if log_x:
        keep &= x > 0
    x, y = x[keep], y[keep]
    displayed_x = np.log10(x) if log_x else x
    result = {"correlation_n": len(x), "spearman_rho": np.nan, "pearson_r": np.nan}
    if len(x) >= 2 and np.unique(displayed_x).size > 1 and np.unique(y).size > 1:
        result["spearman_rho"] = float(spearmanr(x, y).statistic)
        result["pearson_r"] = float(pearsonr(displayed_x, y).statistic)
    return result


def correlation_label(rho, r, log_x=False):
    def number(value):
        return f"{value:.2f}" if np.isfinite(value) else "n/a"
    suffix = r" (log$_{10}$ x)" if log_x else ""
    return (rf"$\rho_{{\mathrm{{S}}}}$ = {number(rho)}; "
            + rf"$r_{{\mathrm{{P}}}}$ = {number(r)}" + suffix)
