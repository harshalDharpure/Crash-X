#!/usr/bin/env python3
"""Bootstrap confidence intervals and paired significance tests for CrashX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats


@dataclass
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    std: float
    n_bootstrap: int

    def format(self, decimals: int = 3, show_ci: bool = True) -> str:
        if show_ci:
            return f"{self.mean:.{decimals}f} [{self.ci_low:.{decimals}f}, {self.ci_high:.{decimals}f}]"
        return f"{self.mean:.{decimals}f}"


@dataclass
class PairedTestResult:
    statistic: float
    p_value: float
    test: str
    n_pairs: int
    significant: bool

    def format_p(self) -> str:
        if self.p_value < 0.001:
            return "<0.001"
        return f"{self.p_value:.3f}"


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    stat_fn: Callable[[np.ndarray], float] | None = None,
) -> BootstrapResult:
    """Bootstrap CI for a 1D array of per-sample scores."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 0.0, n_bootstrap)

    fn = stat_fn or np.mean
    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots[i] = fn(sample)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return BootstrapResult(
        mean=float(fn(arr)),
        ci_low=float(lo),
        ci_high=float(hi),
        std=float(np.std(boots)),
        n_bootstrap=n_bootstrap,
    )


def bootstrap_metric_on_outputs(
    outputs: Sequence[dict[str, Any]],
    metric_fn: Callable[[list[dict[str, Any]]], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Resample videos with replacement and recompute a corpus metric."""
    n = len(outputs)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 0.0, n_bootstrap)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        subset = [outputs[j] for j in idx]
        boots[i] = metric_fn(subset)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
    point = metric_fn(list(outputs))
    return BootstrapResult(
        mean=float(point),
        ci_low=float(lo),
        ci_high=float(hi),
        std=float(np.std(boots)),
        n_bootstrap=n_bootstrap,
    )


def paired_wilcoxon(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> PairedTestResult:
    """Wilcoxon signed-rank test on paired per-video scores."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if va.shape != vb.shape:
        raise ValueError("Paired arrays must have the same length")
    if va.size < 5:
        return PairedTestResult(0.0, 1.0, "wilcoxon", int(va.size), False)

    try:
        stat, p = stats.wilcoxon(va, vb, alternative=alternative, zero_method="wilcox")
    except ValueError:
        stat, p = 0.0, 1.0

    return PairedTestResult(
        statistic=float(stat),
        p_value=float(p),
        test="wilcoxon",
        n_pairs=int(va.size),
        significant=bool(p < 0.05),
    )


def paired_ttest(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> PairedTestResult:
    """Paired t-test on per-video scores."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if va.shape != vb.shape:
        raise ValueError("Paired arrays must have the same length")
    if va.size < 5:
        return PairedTestResult(0.0, 1.0, "ttest", int(va.size), False)

    stat, p = stats.ttest_rel(va, vb, alternative=alternative)
    return PairedTestResult(
        statistic=float(stat),
        p_value=float(p),
        test="ttest",
        n_pairs=int(va.size),
        significant=bool(p < 0.05),
    )
