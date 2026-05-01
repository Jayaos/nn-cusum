import argparse
import json
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines._kernel_mmd import eu_dist2, rbf_kernel_from_dist2


DEFAULT_BANDWIDTH_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)


def load_samples(path):
    samples = np.load(path)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    return samples


def validate_samples(f0_samples, f1_samples):
    if f0_samples.ndim != 2:
        raise ValueError("f0_samples must be a 2D array")
    if f1_samples.ndim != 2:
        raise ValueError("f1_samples must be a 2D array")
    if f0_samples.shape[1] != f1_samples.shape[1]:
        raise ValueError(
            "f0_samples and f1_samples must have the same feature dimension; "
            f"got {f0_samples.shape[1]} and {f1_samples.shape[1]}"
        )
    if f0_samples.shape[0] < 2:
        raise ValueError("f0_samples must contain at least two rows")
    if f1_samples.shape[0] < 2:
        raise ValueError("f1_samples must contain at least two rows")
    if not np.all(np.isfinite(f0_samples)):
        raise ValueError("f0_samples contains NaN or infinite values")
    if not np.all(np.isfinite(f1_samples)):
        raise ValueError("f1_samples contains NaN or infinite values")


def parse_bandwidth_factors(value):
    if value is None:
        return np.asarray(DEFAULT_BANDWIDTH_FACTORS, dtype=float)

    factors = np.fromstring(value, sep=",", dtype=float)
    if factors.size == 0:
        raise ValueError("--bandwidth_factors must contain at least one value")
    if not np.all(np.isfinite(factors)):
        raise ValueError("--bandwidth_factors contains NaN or infinite values")
    if np.any(factors <= 0.0):
        raise ValueError("--bandwidth_factors values must be positive")
    return factors


def shuffled_subset(samples, max_samples, rng):
    indices = rng.permutation(samples.shape[0])
    if max_samples is not None:
        indices = indices[:max_samples]
    return samples[indices]


def standardize_samples(f0_samples, f1_samples):
    combined = np.vstack([f0_samples, f1_samples])
    mean = np.mean(combined, axis=0)
    scale = np.std(combined, axis=0, ddof=0)
    scale[scale == 0.0] = 1.0

    return (
        (f0_samples - mean) / scale,
        (f1_samples - mean) / scale,
        mean,
        scale,
    )


def cross_median_bandwidth(f0_samples, f1_samples, num_median_pairs, rng):
    if num_median_pairs <= 0:
        raise ValueError("num_median_pairs must be positive")

    f0_index = rng.integers(0, f0_samples.shape[0], size=num_median_pairs)
    f1_index = rng.integers(0, f1_samples.shape[0], size=num_median_pairs)
    distances = np.linalg.norm(f0_samples[f0_index] - f1_samples[f1_index], axis=1)
    positive_distances = distances[distances > 0.0]
    if positive_distances.size == 0:
        raise ValueError("median heuristic requires at least one nonzero cross distance")
    return float(np.median(positive_distances))


def off_diagonal_kernel_mean(samples, bandwidth, batch_size):
    total = 0.0
    num_samples = samples.shape[0]

    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        for other_start in range(0, num_samples, batch_size):
            other_stop = min(other_start + batch_size, num_samples)
            dist2 = eu_dist2(samples[start:stop], samples[other_start:other_stop])
            kernel = rbf_kernel_from_dist2(dist2, bandwidth)
            if start == other_start:
                kernel[np.arange(stop - start), np.arange(stop - start)] = 0.0
            total += float(np.sum(kernel))

    return total / (num_samples * (num_samples - 1))


def cross_kernel_mean(f0_samples, f1_samples, bandwidth, batch_size):
    total = 0.0
    for start in range(0, f0_samples.shape[0], batch_size):
        stop = min(start + batch_size, f0_samples.shape[0])
        for other_start in range(0, f1_samples.shape[0], batch_size):
            other_stop = min(other_start + batch_size, f1_samples.shape[0])
            dist2 = eu_dist2(f0_samples[start:stop], f1_samples[other_start:other_stop])
            total += float(np.sum(rbf_kernel_from_dist2(dist2, bandwidth)))
    return total / (f0_samples.shape[0] * f1_samples.shape[0])


def unbiased_mmd2_rbf(f0_samples, f1_samples, bandwidth, batch_size=10000):
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    k00 = off_diagonal_kernel_mean(f0_samples, bandwidth, batch_size)
    k11 = off_diagonal_kernel_mean(f1_samples, bandwidth, batch_size)
    k01 = cross_kernel_mean(f0_samples, f1_samples, bandwidth, batch_size)
    return float(k00 + k11 - 2.0 * k01)


def estimate_mmd2(
    f0_samples,
    f1_samples,
    bandwidth_factors=None,
    num_median_pairs=10000,
    seed=2026,
    max_samples=None,
    standardize=True,
    batch_size=2000,
):
    bandwidth_factors = np.asarray(
        DEFAULT_BANDWIDTH_FACTORS if bandwidth_factors is None else bandwidth_factors,
        dtype=float,
    )
    if np.any(bandwidth_factors <= 0.0):
        raise ValueError("bandwidth_factors values must be positive")
    if num_median_pairs <= 0:
        raise ValueError("num_median_pairs must be positive")
    if max_samples is not None and max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    f0_samples = np.asarray(f0_samples, dtype=float)
    f1_samples = np.asarray(f1_samples, dtype=float)
    if f0_samples.ndim == 1:
        f0_samples = f0_samples.reshape(-1, 1)
    if f1_samples.ndim == 1:
        f1_samples = f1_samples.reshape(-1, 1)
    validate_samples(f0_samples, f1_samples)

    rng = np.random.default_rng(seed)
    f0_eval = shuffled_subset(f0_samples, max_samples, rng)
    f1_eval = shuffled_subset(f1_samples, max_samples, rng)

    standardization = None
    if standardize:
        f0_eval, f1_eval, mean, scale = standardize_samples(f0_eval, f1_eval)
        standardization = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        }

    median_bandwidth = cross_median_bandwidth(
        f0_samples=f0_eval,
        f1_samples=f1_eval,
        num_median_pairs=num_median_pairs,
        rng=rng,
    )
    bandwidth_grid = median_bandwidth * bandwidth_factors

    bandwidth_results = []
    for bandwidth in bandwidth_grid:
        bandwidth_results.append(
            {
                "bandwidth": float(bandwidth),
                "mmd2_unbiased": unbiased_mmd2_rbf(
                    f0_samples=f0_eval,
                    f1_samples=f1_eval,
                    bandwidth=float(bandwidth),
                    batch_size=batch_size,
                ),
            }
        )

    best = max(bandwidth_results, key=lambda item: item["mmd2_unbiased"])
    return {
        "mmd2_unbiased": float(best["mmd2_unbiased"]),
        "selected_bandwidth": float(best["bandwidth"]),
        "median_bandwidth": float(median_bandwidth),
        "bandwidth_factors": bandwidth_factors.tolist(),
        "bandwidth_grid": bandwidth_grid.tolist(),
        "bandwidth_results": bandwidth_results,
        "num_f0_samples": int(f0_eval.shape[0]),
        "num_f1_samples": int(f1_eval.shape[0]),
        "num_median_pairs": int(num_median_pairs),
        "data_dim": int(f0_eval.shape[1]),
        "seed": int(seed),
        "standardize": bool(standardize),
        "standardization": standardization,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate empirical MMD^2 between f0 and f1 samples with an RBF kernel."
    )
    parser.add_argument("--f0_path", type=str, required=True)
    parser.add_argument("--f1_path", type=str, required=True)
    parser.add_argument(
        "--bandwidth_factors",
        type=str,
        default=None,
        help="Comma-separated positive factors applied to the cross-sample median distance.",
    )
    parser.add_argument("--num_median_pairs", type=int, default=10000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save_path", type=str, default=None)

    standardize_group = parser.add_mutually_exclusive_group()
    standardize_group.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        default=True,
    )
    standardize_group.add_argument(
        "--no_standardize",
        dest="standardize",
        action="store_false",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    f0_samples = load_samples(args.f0_path)
    f1_samples = load_samples(args.f1_path)
    bandwidth_factors = parse_bandwidth_factors(args.bandwidth_factors)

    result = estimate_mmd2(
        f0_samples=f0_samples,
        f1_samples=f1_samples,
        bandwidth_factors=bandwidth_factors,
        num_median_pairs=args.num_median_pairs,
        seed=args.seed,
        max_samples=args.max_samples,
        standardize=args.standardize,
        batch_size=args.batch_size,
    )
    result["f0_path"] = os.path.abspath(args.f0_path)
    result["f1_path"] = os.path.abspath(args.f1_path)

    output = json.dumps(result, indent=2)
    print(output)

    if args.save_path is not None:
        save_dir = os.path.dirname(os.path.abspath(args.save_path))
        os.makedirs(save_dir, exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as f:
            f.write(output)
            f.write("\n")
