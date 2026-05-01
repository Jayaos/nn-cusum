import argparse
import json
import math
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.simulation import generate_gamma


def digamma(x):
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0

    inverse = 1.0 / x
    inverse2 = inverse * inverse
    return result + math.log(x) - 0.5 * inverse - inverse2 * (
        1.0 / 12.0
        - inverse2 * (1.0 / 120.0 - inverse2 * (1.0 / 252.0))
    )


def shifted_gamma_logpdf(x, shape, scale, location_shift=0.0):
    x = np.asarray(x, dtype=float)
    y = x - location_shift
    logpdf = np.full_like(x, -np.inf, dtype=float)
    support = y > 0.0

    logpdf[support] = (
        (shape - 1.0) * np.log(y[support])
        - y[support] / scale
        - shape * np.log(scale)
        - math.lgamma(shape)
    )
    return np.sum(logpdf, axis=1)


def analytic_kl_q_p_same_location(
    dim,
    p_shape,
    p_scale,
    q_shape,
    q_scale,
):
    one_dim_kl = (
        (q_shape - p_shape) * digamma(q_shape)
        - math.lgamma(q_shape)
        + math.lgamma(p_shape)
        + p_shape * math.log(p_scale / q_scale)
        + q_shape * (q_scale / p_scale - 1.0)
    )
    return float(dim * one_dim_kl)


def estimate_kl_gamma_q_p(
    dim,
    p_shape,
    p_scale,
    q_shape,
    q_scale,
    q_loc_shift,
    num_samples=200000,
    seed=2026,
    p_loc_shift=0.0,
    batch_size=50000,
):
    values = []
    samples_remaining = num_samples
    batch_index = 0

    while samples_remaining > 0:
        current_batch_size = min(batch_size, samples_remaining)
        x = generate_gamma(
            shape=q_shape,
            scale=q_scale,
            dim=dim,
            n=current_batch_size,
            location_shift=q_loc_shift,
            seed=seed + batch_index,
        )

        log_q = shifted_gamma_logpdf(
            x,
            shape=q_shape,
            scale=q_scale,
            location_shift=q_loc_shift,
        )
        log_p = shifted_gamma_logpdf(
            x,
            shape=p_shape,
            scale=p_scale,
            location_shift=p_loc_shift,
        )
        values.append(log_q - log_p)

        samples_remaining -= current_batch_size
        batch_index += 1

    kl_values = np.concatenate(values)
    kl_estimate = float(np.mean(kl_values))
    standard_error = float(np.std(kl_values, ddof=1) / np.sqrt(num_samples))

    analytic_kl = None
    if q_loc_shift == p_loc_shift:
        analytic_kl = analytic_kl_q_p_same_location(
            dim=dim,
            p_shape=p_shape,
            p_scale=p_scale,
            q_shape=q_shape,
            q_scale=q_scale,
        )

    return {
        "kl_q_p": kl_estimate,
        "standard_error": standard_error,
        "analytic_kl_q_p_same_location": analytic_kl,
        "num_samples": int(num_samples),
        "data_dim": int(dim),
        "seed": int(seed),
        "p_shape": float(p_shape),
        "p_scale": float(p_scale),
        "p_loc_shift": float(p_loc_shift),
        "q_shape": float(q_shape),
        "q_scale": float(q_scale),
        "q_loc_shift": float(q_loc_shift),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate KL(q || p) for shifted gamma distributions."
    )
    parser.add_argument("--data_dim", type=int, required=True)
    parser.add_argument("--p_shape", type=float, required=True)
    parser.add_argument("--p_scale", type=float, required=True)
    parser.add_argument("--q_shape", type=float, required=True)
    parser.add_argument("--q_scale", type=float, required=True)
    parser.add_argument("--q_loc_shift", type=float, required=True)
    parser.add_argument("--p_loc_shift", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
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
    if args.p_shape <= 0.0:
        raise ValueError("--p_shape must be positive")
    if args.p_scale <= 0.0:
        raise ValueError("--p_scale must be positive")
    if args.q_shape <= 0.0:
        raise ValueError("--q_shape must be positive")
    if args.q_scale <= 0.0:
        raise ValueError("--q_scale must be positive")
    if args.q_loc_shift < args.p_loc_shift:
        raise ValueError(
            "KL(q || p) is infinite when q support starts before p support; "
            "require --q_loc_shift >= --p_loc_shift"
        )

    result = estimate_kl_gamma_q_p(
        dim=args.data_dim,
        p_shape=args.p_shape,
        p_scale=args.p_scale,
        q_shape=args.q_shape,
        q_scale=args.q_scale,
        q_loc_shift=args.q_loc_shift,
        num_samples=args.num_samples,
        seed=args.seed,
        p_loc_shift=args.p_loc_shift,
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
