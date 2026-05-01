import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


DEFAULT_BANDWIDTH_GRID = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


def load_samples(path):
    samples = np.load(path)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    return samples


def validate_samples(p_samples, q_samples, cv_folds):
    if p_samples.ndim != 2:
        raise ValueError("p_samples must be a 2D array")
    if q_samples.ndim != 2:
        raise ValueError("q_samples must be a 2D array")
    if p_samples.shape[1] != q_samples.shape[1]:
        raise ValueError(
            "p_samples and q_samples must have the same feature dimension; "
            f"got {p_samples.shape[1]} and {q_samples.shape[1]}"
        )
    if p_samples.shape[0] < cv_folds:
        raise ValueError("p_samples must contain at least cv_folds rows")
    if q_samples.shape[0] <= cv_folds:
        raise ValueError(
            "q_samples must contain more than cv_folds rows so that held-out "
            "evaluation samples are available"
        )
    if not np.all(np.isfinite(p_samples)):
        raise ValueError("p_samples contains NaN or infinite values")
    if not np.all(np.isfinite(q_samples)):
        raise ValueError("q_samples contains NaN or infinite values")


def parse_bandwidth_grid(value):
    if value is None:
        return np.asarray(DEFAULT_BANDWIDTH_GRID, dtype=float)

    bandwidths = np.fromstring(value, sep=",", dtype=float)
    if bandwidths.size == 0:
        raise ValueError("--bandwidth_grid must contain at least one value")
    if not np.all(np.isfinite(bandwidths)):
        raise ValueError("--bandwidth_grid contains NaN or infinite values")
    if np.any(bandwidths <= 0.0):
        raise ValueError("--bandwidth_grid values must be positive")
    return bandwidths


def shuffled_subset(samples, max_samples, rng):
    indices = rng.permutation(samples.shape[0])
    if max_samples is not None:
        indices = indices[:max_samples]
    return samples[indices]


def split_train_eval_q(q_samples, eval_fraction, max_train_samples, max_eval_samples, rng):
    indices = rng.permutation(q_samples.shape[0])
    num_eval = int(np.floor(q_samples.shape[0] * eval_fraction))
    num_eval = max(1, num_eval)
    if max_eval_samples is not None:
        num_eval = min(num_eval, max_eval_samples)
    num_eval = min(num_eval, q_samples.shape[0] - 1)

    q_eval = q_samples[indices[:num_eval]]
    q_train = q_samples[indices[num_eval:]]
    if max_train_samples is not None:
        q_train = q_train[:max_train_samples]
    return q_train, q_eval


def standardize_train_eval(p_train, q_train, q_eval):
    combined_train = np.vstack([p_train, q_train])
    mean = np.mean(combined_train, axis=0)
    scale = np.std(combined_train, axis=0, ddof=0)
    scale[scale == 0.0] = 1.0

    return (
        (p_train - mean) / scale,
        (q_train - mean) / scale,
        (q_eval - mean) / scale,
        mean,
        scale,
    )


def select_kde_bandwidth(samples, bandwidth_grid, cv_folds, seed):
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_results = []

    for bandwidth in bandwidth_grid:
        fold_scores = []
        for train_index, val_index in kfold.split(samples):
            kde = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))
            kde.fit(samples[train_index])
            fold_scores.append(float(np.mean(kde.score_samples(samples[val_index]))))

        cv_results.append(
            {
                "bandwidth": float(bandwidth),
                "mean_log_likelihood": float(np.mean(fold_scores)),
                "std_log_likelihood": float(np.std(fold_scores, ddof=1))
                if len(fold_scores) > 1
                else 0.0,
            }
        )

    best = max(cv_results, key=lambda item: item["mean_log_likelihood"])
    return best["bandwidth"], cv_results


def score_samples_in_batches(kde, samples, batch_size):
    scores = []
    for start in range(0, samples.shape[0], batch_size):
        stop = min(start + batch_size, samples.shape[0])
        scores.append(kde.score_samples(samples[start:stop]))
    return np.concatenate(scores)


def estimate_kl_kde_q_p(
    p_samples,
    q_samples,
    bandwidth_grid=None,
    cv_folds=5,
    seed=2026,
    eval_fraction=0.2,
    max_train_samples=None,
    max_eval_samples=None,
    standardize=True,
    batch_size=10000,
):
    bandwidth_grid = np.asarray(
        DEFAULT_BANDWIDTH_GRID if bandwidth_grid is None else bandwidth_grid,
        dtype=float,
    )
    if np.any(bandwidth_grid <= 0.0):
        raise ValueError("bandwidth_grid values must be positive")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    if max_train_samples is not None and max_train_samples < cv_folds:
        raise ValueError("max_train_samples must be at least cv_folds")
    if max_eval_samples is not None and max_eval_samples < 1:
        raise ValueError("max_eval_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    p_samples = np.asarray(p_samples, dtype=float)
    q_samples = np.asarray(q_samples, dtype=float)
    if p_samples.ndim == 1:
        p_samples = p_samples.reshape(-1, 1)
    if q_samples.ndim == 1:
        q_samples = q_samples.reshape(-1, 1)
    validate_samples(p_samples, q_samples, cv_folds)

    rng = np.random.default_rng(seed)
    p_train = shuffled_subset(p_samples, max_train_samples, rng)
    q_train, q_eval = split_train_eval_q(
        q_samples=q_samples,
        eval_fraction=eval_fraction,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        rng=rng,
    )

    if p_train.shape[0] < cv_folds:
        raise ValueError("p_train contains fewer than cv_folds samples")
    if q_train.shape[0] < cv_folds:
        raise ValueError("q_train contains fewer than cv_folds samples")

    standardization = None
    if standardize:
        p_train, q_train, q_eval, mean, scale = standardize_train_eval(
            p_train=p_train,
            q_train=q_train,
            q_eval=q_eval,
        )
        standardization = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        }

    p_bandwidth, p_cv_results = select_kde_bandwidth(
        samples=p_train,
        bandwidth_grid=bandwidth_grid,
        cv_folds=cv_folds,
        seed=seed,
    )
    q_bandwidth, q_cv_results = select_kde_bandwidth(
        samples=q_train,
        bandwidth_grid=bandwidth_grid,
        cv_folds=cv_folds,
        seed=seed + 1,
    )

    p_kde = KernelDensity(kernel="gaussian", bandwidth=p_bandwidth)
    q_kde = KernelDensity(kernel="gaussian", bandwidth=q_bandwidth)
    p_kde.fit(p_train)
    q_kde.fit(q_train)

    log_q = score_samples_in_batches(q_kde, q_eval, batch_size)
    log_p = score_samples_in_batches(p_kde, q_eval, batch_size)
    kl_values = log_q - log_p

    return {
        "kl_q_p": float(np.mean(kl_values)),
        "standard_error": float(np.std(kl_values, ddof=1) / np.sqrt(q_eval.shape[0]))
        if q_eval.shape[0] > 1
        else 0.0,
        "p_bandwidth": float(p_bandwidth),
        "q_bandwidth": float(q_bandwidth),
        "bandwidth_grid": bandwidth_grid.tolist(),
        "p_cv_results": p_cv_results,
        "q_cv_results": q_cv_results,
        "cv_folds": int(cv_folds),
        "num_p_train_samples": int(p_train.shape[0]),
        "num_q_train_samples": int(q_train.shape[0]),
        "num_q_eval_samples": int(q_eval.shape[0]),
        "data_dim": int(p_train.shape[1]),
        "seed": int(seed),
        "eval_fraction": float(eval_fraction),
        "standardize": bool(standardize),
        "standardization": standardization,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate KL(q || p) from real samples using Gaussian KDE."
    )
    parser.add_argument("--p_path", type=str, required=True)
    parser.add_argument("--q_path", type=str, required=True)
    parser.add_argument(
        "--bandwidth_grid",
        type=str,
        default=None,
        help="Comma-separated positive bandwidth candidates. Defaults to a fixed grid.",
    )
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=10000)
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

    p_samples = load_samples(args.p_path)
    q_samples = load_samples(args.q_path)
    bandwidth_grid = parse_bandwidth_grid(args.bandwidth_grid)

    result = estimate_kl_kde_q_p(
        p_samples=p_samples,
        q_samples=q_samples,
        bandwidth_grid=bandwidth_grid,
        cv_folds=args.cv_folds,
        seed=args.seed,
        eval_fraction=args.eval_fraction,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        standardize=args.standardize,
        batch_size=args.batch_size,
    )
    result["p_path"] = os.path.abspath(args.p_path)
    result["q_path"] = os.path.abspath(args.q_path)

    output = json.dumps(result, indent=2)
    print(output)

    if args.save_path is not None:
        save_dir = os.path.dirname(os.path.abspath(args.save_path))
        os.makedirs(save_dir, exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as f:
            f.write(output)
            f.write("\n")
