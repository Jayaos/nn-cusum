import os
import sys

import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from run_analysis.compute_mmd import estimate_mmd2, unbiased_mmd2_rbf


def test_mmd_uses_median_factor_grid_and_selects_from_it():
    rng = np.random.default_rng(123)
    f0_samples = rng.normal(loc=0.0, scale=1.0, size=(60, 2))
    f1_samples = rng.normal(loc=0.7, scale=1.0, size=(60, 2))
    bandwidth_factors = np.array([0.5, 1.0, 2.0])

    result = estimate_mmd2(
        f0_samples=f0_samples,
        f1_samples=f1_samples,
        bandwidth_factors=bandwidth_factors,
        num_median_pairs=200,
        seed=123,
        max_samples=50,
        batch_size=17,
    )

    expected_grid = result["median_bandwidth"] * bandwidth_factors
    assert np.allclose(result["bandwidth_grid"], expected_grid)
    assert result["selected_bandwidth"] in result["bandwidth_grid"]
    assert result["mmd2_unbiased"] == max(
        item["mmd2_unbiased"] for item in result["bandwidth_results"]
    )
    assert np.isfinite(result["mmd2_unbiased"])
    assert result["num_f0_samples"] == 50
    assert result["num_f1_samples"] == 50


def test_unbiased_mmd2_matches_direct_formula():
    f0_samples = np.array([[0.0], [1.0], [2.0]])
    f1_samples = np.array([[1.5], [2.5]])
    bandwidth = 1.25

    k00 = np.exp(-((f0_samples - f0_samples.T) ** 2) / (2.0 * bandwidth**2))
    k11 = np.exp(-((f1_samples - f1_samples.T) ** 2) / (2.0 * bandwidth**2))
    k01 = np.exp(-((f0_samples - f1_samples.T) ** 2) / (2.0 * bandwidth**2))
    expected = (
        (np.sum(k00) - np.trace(k00)) / (3 * 2)
        + (np.sum(k11) - np.trace(k11)) / (2 * 1)
        - 2.0 * np.mean(k01)
    )

    actual = unbiased_mmd2_rbf(
        f0_samples=f0_samples,
        f1_samples=f1_samples,
        bandwidth=bandwidth,
        batch_size=2,
    )

    assert actual == pytest.approx(expected)


def test_mmd_rejects_dimension_mismatch():
    f0_samples = np.zeros((20, 2))
    f1_samples = np.zeros((20, 3))

    with pytest.raises(ValueError, match="same feature dimension"):
        estimate_mmd2(
            f0_samples=f0_samples,
            f1_samples=f1_samples,
            num_median_pairs=20,
        )
