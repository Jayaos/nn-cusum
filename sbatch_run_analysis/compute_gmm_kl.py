import argparse
import json
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.simulation import generate_gaussian_mixture_q


def parse_mu3(mu3_arg, dim):
    if mu3_arg is None or str(mu3_arg).lower() == "none":
        return np.zeros(dim, dtype=float)

    values = np.fromstring(mu3_arg, sep=",", dtype=float)
    if values.size == 1:
        return np.full(dim, values.item(), dtype=float)
    if values.size != dim:
        raise ValueError(
            f"--mu3 must be a scalar or a comma-separated vector of length {dim}; "
            f"got length {values.size}"
        )
    return values


def logsumexp(values, axis=1):
    max_values = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_values, axis=axis) + np.log(
        np.sum(np.exp(values - max_values), axis=axis)
    )


def gaussian_logpdf(x, mean, cov):
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    dim = mean.size
    chol = np.linalg.cholesky(cov)
    centered = (x - mean).T
    solved = np.linalg.solve(chol, centered)
    mahalanobis = np.sum(solved * solved, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))

    return -0.5 * (dim * np.log(2.0 * np.pi) + log_det + mahalanobis)


def gaussian_mixture_logpdf(x, weights, means, covs):
    component_logpdfs = []
    for weight, mean, cov in zip(weights, means, covs):
        component_logpdfs.append(np.log(weight) + gaussian_logpdf(x, mean, cov))
    return logsumexp(np.column_stack(component_logpdfs), axis=1)


def build_gaussian_mixture_params(dim, rho=None, mu3=None):
    if rho is None:
        rho = 0.2
    if mu3 is None:
        mu3 = np.zeros(dim, dtype=float)

    one = np.ones(dim, dtype=float)
    identity = np.eye(dim, dtype=float)
    all_ones = np.ones((dim, dim), dtype=float)

    p_weights = np.array([0.5, 0.5], dtype=float)
    p_means = [2.0 * one, -2.0 * one]
    p_covs = [identity, identity]

    diag = np.sqrt(rho) * identity
    sigma3 = identity - diag @ diag + diag @ all_ones @ diag

    q_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    q_means = [2.0 * one, -2.0 * one, mu3]
    q_covs = [identity, identity, sigma3]

    return (p_weights, p_means, p_covs), (q_weights, q_means, q_covs), diag


def estimate_kl_gmm_q_p(
    dim,
    num_samples,
    seed=2026,
    rho=None,
    mu3=None,
    batch_size=50000,
):
    if rho is None:
        rho = 0.2
    if mu3 is None:
        mu3 = np.zeros(dim, dtype=float)
    else:
        mu3 = np.asarray(mu3, dtype=float)

    p_params, q_params, diag = build_gaussian_mixture_params(dim, rho, mu3)

    values = []
    samples_remaining = num_samples
    batch_index = 0

    while samples_remaining > 0:
        current_batch_size = min(batch_size, samples_remaining)
        x = generate_gaussian_mixture_q(
            dim=dim,
            n=current_batch_size,
            seed=seed + batch_index,
            mu3=mu3,
            diag=diag,
        )

        log_q = gaussian_mixture_logpdf(x, *q_params)
        log_p = gaussian_mixture_logpdf(x, *p_params)
        values.append(log_q - log_p)

        samples_remaining -= current_batch_size
        batch_index += 1

    kl_values = np.concatenate(values)
    kl_estimate = float(np.mean(kl_values))
    standard_error = float(np.std(kl_values, ddof=1) / np.sqrt(num_samples))

    return {
        "kl_q_p": kl_estimate,
        "standard_error": standard_error,
        "num_samples": int(num_samples),
        "data_dim": int(dim),
        "seed": int(seed),
        "rho": float(rho),
        "mu3": mu3.tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate KL(q || p) for the Gaussian mixture in utils.simulation."
    )
    parser.add_argument("--data_dim", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Correlation parameter for Sigma3 = (1-rho) I + rho E.",
    )
    parser.add_argument(
        "--mu3",
        type=str,
        default=None,
        help="Third component mean: scalar or comma-separated vector. Defaults to zero.",
    )
    parser.add_argument("--save_path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.data_dim <= 0:
        raise ValueError("--data_dim must be positive")
    if args.num_samples <= 1:
        raise ValueError("--num_samples must be greater than 1")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.rho is not None and not 0.0 <= args.rho < 1.0:
        raise ValueError("--rho must satisfy 0 <= rho < 1")

    mu3 = parse_mu3(args.mu3, args.data_dim)
    result = estimate_kl_gmm_q_p(
        dim=args.data_dim,
        num_samples=args.num_samples,
        seed=args.seed,
        rho=args.rho,
        mu3=mu3,
        batch_size=args.batch_size,
    )

    output = json.dumps(result, indent=2)
    print(output)

    if args.save_path is not None:
        save_dir = os.path.dirname(os.path.abspath(args.save_path))
        os.makedirs(save_dir, exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as f:
            f.write(output)
            f.write("\n")
