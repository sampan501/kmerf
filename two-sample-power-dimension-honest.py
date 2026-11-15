#!/usr/bin/env python
# coding: utf-8

# Power vs. Dimension for honest (sample-split) KMERF


import os

import numpy as np
from joblib import Parallel, delayed
from scipy.stats.distributions import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances

from hyppo.independence._utils import sim_matrix
from hyppo.independence.dcorr import _dcorr
from simulations import make_marron_wand_classification, MARRON_WAND_SIMS


DIMENSIONS = range(3, 11)
SAMP_SIZE = 400
REPS = range(1000)
EST_NAME = "KMERF-Honest"


SAVE_PATH = "two-sample-n-{}_p-{}_{}".format(
    int(SAMP_SIZE), int(DIMENSIONS[0]), int(DIMENSIONS[-1])
)
os.makedirs(SAVE_PATH, exist_ok=True)


def compute_pval(rep, sim, p):
    """
    Fit forest on a random half, chi-square p-value (as KMERF.test) on the other half.
    """
    out_file = f"{SAVE_PATH}/{sim}_{EST_NAME}_{p}_{rep}.txt"
    if os.path.exists(out_file):
        return
    X, _ = make_marron_wand_classification(
        n_samples=SAMP_SIZE,
        n_dim=DIMENSIONS[-1],
        n_informative=1,
        simulation=sim,
        seed=rep,
    )
    X = X[:, :p]
    y = np.concatenate((np.zeros(SAMP_SIZE // 2), np.ones(SAMP_SIZE // 2)))
    idx = np.random.default_rng(rep).permutation(SAMP_SIZE)
    a, b = idx[: SAMP_SIZE // 2], idx[SAMP_SIZE // 2 :]
    forest = RandomForestClassifier(n_estimators=500).fit(X[a], y[a])
    distx = 1 - sim_matrix(forest, X[b])
    disty = pairwise_distances(y[b].reshape(-1, 1))
    stat = _dcorr(distx, disty, bias=False, is_fast=False)
    pval = chi2.sf(len(b) * stat + 1, 1)
    np.savetxt(out_file, [pval])


_ = Parallel(n_jobs=8, verbose=100)(
    [
        delayed(compute_pval)(rep, sim, p=dim)
        for rep in REPS
        for sim in MARRON_WAND_SIMS.keys()
        for dim in DIMENSIONS
    ]
)
