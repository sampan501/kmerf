"""Null calibration of KMERF.test()'s chi-square p-value under five forest-fitting arms."""

import numpy as np
from joblib import Parallel, delayed
from scipy.stats.distributions import chi2
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import pairwise_distances

import hyppo
from hyppo.independence._utils import sim_matrix
from hyppo.independence.dcorr import _dcorr

print(hyppo.__file__, hyppo.__version__, flush=True)

N, REPS = 100, 1000


def pval(forest, X, y):
    # same as KMERF.test(): _dcorr is already normalized, so statx = staty = 1
    distx = 1 - sim_matrix(forest, X)
    disty = pairwise_distances(y.reshape(-1, 1))
    stat = _dcorr(distx, disty, bias=False, is_fast=False)
    return chi2.sf(len(y) * stat + 1, 1)


def one_rep(rep, arm):
    rng = np.random.default_rng(rep)
    X = rng.normal(size=(N, 3))
    y = rng.normal(size=N)
    rf = RandomForestRegressor(n_estimators=500, random_state=rep)
    if arm == "a":
        return pval(rf.fit(X, y), X, y)
    if arm == "b":
        rf.set_params(min_samples_leaf=5)
        return pval(rf.fit(X, y), X, y)
    if arm == "c":
        y = np.repeat([0.0, 1.0], N // 2)
        clf = RandomForestClassifier(n_estimators=500, random_state=rep)
        return pval(clf.fit(X, y), X, y)
    if arm == "d":
        return pval(rf.fit(X, rng.normal(size=N)), X, y)
    if arm == "e":
        return pval(rf.fit(X[:50], y[:50]), X[50:], y[50:])


ARMS = {
    "a": "trained on (X, y), y ~ N(0,1), default leaf",
    "b": "trained on (X, y), min_samples_leaf=5",
    "c": "classifier trained on (X, y), y binary 0/1",
    "d": "trained on unrelated N(0,1) response",
    "e": "trained on rows :50, p-value on rows 50:",
}
for arm, desc in ARMS.items():
    p = np.array(Parallel(n_jobs=2)(delayed(one_rep)(r, arm) for r in range(REPS)))
    print(f"{arm}. {desc:45s} P(p <= 0.05) = {(p <= 0.05).mean():.3f}", flush=True)
