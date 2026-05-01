import os
import sys

import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from run_analysis.compute_kde_kl import estimate_kl_kde_q_p


def test_kde_kl_selects_bandwidth_from_grid():
    rng = np.random.default_rng(123)
    p_samples = rng.normal(loc=0.0, scale=1.0, size=(80, 2))
    q_samples = rng.normal(loc=0.5, scale=1.0, size=(80, 2))
    bandwidth_grid = np.array([0.2, 0.5, 1.0])

    result = estimate_kl_kde_q_p(
        p_samples=p_samples,
        q_samples=q_samples,
        bandwidth_grid=bandwidth_grid,
        cv_folds=5,
        seed=123,
        max_train_samples=60,
        max_eval_samples=20,
    )

    assert result["p_bandwidth"] in bandwidth_grid
    assert result["q_bandwidth"] in bandwidth_grid
    assert np.isfinite(result["kl_q_p"])
    assert result["num_q_eval_samples"] == 16


def test_kde_kl_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(456)
    samples = rng.normal(loc=0.0, scale=1.0, size=(100, 1))

    result = estimate_kl_kde_q_p(
        p_samples=samples,
        q_samples=samples,
        bandwidth_grid=np.array([0.2, 0.5, 1.0]),
        cv_folds=5,
        seed=456,
        max_train_samples=80,
        max_eval_samples=20,
    )

    assert abs(result["kl_q_p"]) < 0.5


def test_kde_kl_rejects_dimension_mismatch():
    p_samples = np.zeros((20, 2))
    q_samples = np.zeros((20, 3))

    with pytest.raises(ValueError, match="same feature dimension"):
        estimate_kl_kde_q_p(
            p_samples=p_samples,
            q_samples=q_samples,
            bandwidth_grid=np.array([0.5, 1.0]),
            cv_folds=5,
        )
