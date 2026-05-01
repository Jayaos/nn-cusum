import argparse
import json
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.simulation import generate_exponential


def shifted_exponential_logpdf(x, beta, mu):
    x = np.asarray(x, dtype=float)
    logpdf = np.full_like(x, -np.inf, dtype=float)
    support = x >= mu
    logpdf[support] = -np.log(beta) - (x[support] - mu) / beta
    return np.sum(logpdf, axis=1)


def analytic_kl_q_p(
    dim,
    p_beta=1.0,
    p_mu=0.0,
    q_beta=0.8,
    q_mu=0.2,
):
    if q_mu < p_mu:
        return np.inf

    one_dim_kl = (
        np.log(p_beta / q_beta)
        - 1.0
        + (q_mu - p_mu + q_beta) / p_beta
    )
    return float(dim * one_dim_kl)


def estimate_kl_exponential_q_p(
    dim,
    num_samples=200000,
    seed=2026,
    p_beta=1.0,
    p_mu=0.0,
    q_beta=0.8,
    q_mu=0.2,
    batch_size=50000,
):
    values = []
    samples_remaining = num_samples
    batch_index = 0

    while samples_remaining > 0:
        current_batch_size = min(batch_size, samples_remaining)
        x = generate_exponential(
            dim=dim,
            n=current_batch_size,
            beta=q_beta,
            mu=q_mu,
            seed=seed + batch_index,
        )

        log_q = shifted_exponential_logpdf(x, beta=q_beta, mu=q_mu)
        log_p = shifted_exponential_logpdf(x, beta=p_beta, mu=p_mu)
        values.append(log_q - log_p)

        samples_remaining -= current_batch_size
        batch_index += 1

    kl_values = np.concatenate(values)
    kl_estimate = float(np.mean(kl_values))
    standard_error = float(np.std(kl_values, ddof=1) / np.sqrt(num_samples))

    return {
        "kl_q_p": kl_estimate,
        "standard_error": standard_error,
        "analytic_kl_q_p": analytic_kl_q_p(
            dim=dim,
            p_beta=p_beta,
            p_mu=p_mu,
            q_beta=q_beta,
            q_mu=q_mu,
        ),
        "num_samples": int(num_samples),
        "data_dim": int(dim),
        "seed": int(seed),
        "p_beta": float(p_beta),
        "p_mu": float(p_mu),
        "q_beta": float(q_beta),
        "q_mu": float(q_mu),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute KL(q || p) for shifted exponential distributions."
    )
    parser.add_argument("--data_dim", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--p_beta", type=float, default=1.0)
    parser.add_argument("--p_mu", type=float, default=0.0)
    parser.add_argument("--q_beta", type=float, default=0.8)
    parser.add_argument("--q_mu", type=float, default=0.2)
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
    if args.p_beta <= 0.0:
        raise ValueError("--p_beta must be positive")
    if args.q_beta <= 0.0:
        raise ValueError("--q_beta must be positive")

    result = estimate_kl_exponential_q_p(
        dim=args.data_dim,
        num_samples=args.num_samples,
        seed=args.seed,
        p_beta=args.p_beta,
        p_mu=args.p_mu,
        q_beta=args.q_beta,
        q_mu=args.q_mu,
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
