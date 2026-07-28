"""
Shared analysis utilities for the non-termination study.

Centralizes the primitives that were previously copy-pasted across scripts
(auroc, within-stratum z-scoring, proportion CIs, bootstrap, the non-termination
predicate). Pure, dependency-light (numpy only; scipy optional for exact tests),
and seeded for reproducibility.

New scripts should import from here:
    from analysis_utils import auroc, nonterm, wilson, zscore_within, boot_ci
Legacy scripts keep their inline copies so their cached outputs stay bit-reproducible.
"""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np

DEFAULT_CAP = 8192


# ---------- IO ----------

def load_jsonl(path):
    """Read a .jsonl file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- non-termination predicate ----------

def nonterm(rec, cap=DEFAULT_CAP):
    """A run is non-terminating when it exhausts its budget with no committed answer:
    empty final answer AND completion tokens within 8 of the cap. Gold-independent."""
    return (rec.get("full_ans") in (None, "")
            and rec.get("completion_tokens", 0) >= cap - 8)


# ---------- ranking / detection ----------

def auroc(scores, labels):
    """Area under ROC via the rank-sum identity. Returns nan if one class is absent.
    Higher `scores` are predicted to be the positive (label==1) class."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels)
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def zscore_within(recs, field):
    """Z-score `field` within each (dataset, condition) stratum, so a downstream score
    cannot exploit 'hard datasets are hard'. Missing values are treated as 0."""
    strata = defaultdict(list)
    for i, r in enumerate(recs):
        strata[(r["dataset"], r["condition"])].append(i)
    z = np.zeros(len(recs))
    for _, idxs in strata.items():
        v = np.array([float(recs[i].get(field) or 0) for i in idxs])
        mu, sd = v.mean(), v.std() + 1e-9
        for j, i in enumerate(idxs):
            z[i] = (v[j] - mu) / sd
    return z


# ---------- interval estimates ----------

def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Robust near 0/1, unlike the
    normal approximation. Returns (point, lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, center - half), min(1.0, center + half)


def boot_ci(values, stat=np.mean, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a 1-D statistic. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    boots = [stat(values[rng.integers(0, n, n)]) for _ in range(n_boot)]
    boots = np.sort(boots)
    return (float(stat(values)),
            float(boots[int(alpha / 2 * n_boot)]),
            float(boots[int((1 - alpha / 2) * n_boot)]))


def auroc_ci(scores, labels, n_boot=2000, seed=0, alpha=0.05):
    """Bootstrap CI for AUROC; degenerate resamples (one class) are skipped.
    Returns (point, lo, hi); (point, nan, nan) if no valid resample exists."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, float)
    labels = np.asarray(labels)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auroc(scores[idx], labels[idx])
        if not np.isnan(a):
            vals.append(a)
    point = auroc(scores, labels)
    if not vals:
        return point, float("nan"), float("nan")
    vals.sort()
    return (point,
            vals[int(alpha / 2 * len(vals))],
            vals[int((1 - alpha / 2) * len(vals))])
