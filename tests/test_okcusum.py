from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from okcusum import median_heuristic_bandwidth, online_kernel_cusum_statistic, prepare_okcusum_reference


REPO_ROOT = Path(__file__).resolve().parents[1]
MATLAB_REFERENCE_CSV = REPO_ROOT / "online_kernel_cusum" / "raw_pre_change_sample_dim20.csv"

# By default, the mixture parameters mirror the MATLAB EDD-vs-ARL reproduction
# script in `online_kernel_cusum/example2_EDDvsARL.m`:
# q = 0.3 N(0, I) + 0.7 N(0, 4I).

_WORKER_STATE: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the leftmost EDD-vs-log10(ARL) curve from Figure 7 / the matching "
            "MATLAB EDD-vs-ARL example using OKCUSUM and Scan-B."
        )
    )
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "results" / "okcusum_figure7_left")
    parser.add_argument("--num_h0_trials", type=int, default=1000)
    parser.add_argument("--num_h1_trials", type=int, default=1000)
    parser.add_argument("--calibration_horizon", type=int, default=2000)
    parser.add_argument("--sample_size", type=int, default=50)
    parser.add_argument("--sample_dim", type=int, default=20)
    parser.add_argument("--num_reference", type=int, default=10000)
    parser.add_argument("--num_blocks", type=int, default=15)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--mix_p", type=float, default=0.3)
    parser.add_argument("--mean1", type=float, default=0.0)
    parser.add_argument("--std1", type=float, default=1.0)
    parser.add_argument("--mean2", type=float, default=0.0)
    parser.add_argument("--std2", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument(
        "--target_arl",
        type=float,
        nargs="+",
        default=[10**2.5, 10**3.0, 10**3.5, 10**4.0, 10**4.5, 10**5.0],
    )
    return parser.parse_args()


def load_reference_sample(num_reference: int, sample_dim: int, seed: int) -> np.ndarray:
    if MATLAB_REFERENCE_CSV.exists():
        reference = np.loadtxt(MATLAB_REFERENCE_CSV, delimiter=",", dtype=np.float64)
        if reference.shape[0] < num_reference or reference.shape[1] != sample_dim:
            raise ValueError(
                "The MATLAB reference sample does not match the requested shape: "
                f"expected at least ({num_reference}, {sample_dim}), found {reference.shape}."
            )
        return reference[:num_reference]

    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(num_reference, sample_dim))


def sample_post_change_mixture(
    rng: np.random.Generator,
    sample_size: int,
    sample_dim: int,
    mix_p: float,
    mean1: float,
    std1: float,
    mean2: float,
    std2: float,
) -> np.ndarray:
    comp1 = rng.normal(loc=mean1, scale=std1, size=(sample_size, sample_dim))
    comp2 = rng.normal(loc=mean2, scale=std2, size=(sample_size, sample_dim))
    choose_comp1 = rng.binomial(1, mix_p, size=sample_size).astype(bool)
    samples = comp2
    samples[choose_comp1] = comp1[choose_comp1]
    return samples


def detection_delay(stat_seq: np.ndarray, threshold: float) -> float:
    hits = np.flatnonzero(stat_seq > threshold)
    return float(np.inf) if hits.size == 0 else float(hits[0] + 1)


def _init_h0_worker(
    pre_change_sample: np.ndarray,
    prepared_reference: dict[str, np.ndarray | float | int],
    calibration_horizon: int,
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {
        "pre_change_sample": pre_change_sample,
        "prepared_reference": prepared_reference,
        "calibration_horizon": calibration_horizon,
    }


def _run_h0_batch(batch_start: int, batch_size: int, seed: int) -> tuple[int, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pre_change_sample = np.asarray(_WORKER_STATE["pre_change_sample"], dtype=np.float64)
    prepared_reference = _WORKER_STATE["prepared_reference"]
    calibration_horizon = int(_WORKER_STATE["calibration_horizon"])

    okcusum_max = np.zeros(batch_size, dtype=np.float64)
    scanb_max = np.zeros(batch_size, dtype=np.float64)

    for local_idx in range(batch_size):
        post_change_sample = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(calibration_horizon, pre_change_sample.shape[1]),
        )
        okcusum_stat, scanb_stat = online_kernel_cusum_statistic(
            pre_change_sample=pre_change_sample,
            post_change_sample=post_change_sample,
            omega_B=np.asarray(prepared_reference["omega_B"], dtype=int),
            num_blocks=int(prepared_reference["num_blocks"]),
            kernel_bandwidth=float(prepared_reference["kernel_bandwidth"]),
            prepared_reference=prepared_reference,
        )
        okcusum_max[local_idx] = np.max(okcusum_stat)
        scanb_max[local_idx] = np.max(scanb_stat)

    return batch_start, okcusum_max, scanb_max


def _init_h1_worker(
    pre_change_sample: np.ndarray,
    prepared_reference: dict[str, np.ndarray | float | int],
    sample_size: int,
    mix_p: float,
    mean1: float,
    std1: float,
    mean2: float,
    std2: float,
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {
        "pre_change_sample": pre_change_sample,
        "prepared_reference": prepared_reference,
        "sample_size": sample_size,
        "mix_p": mix_p,
        "mean1": mean1,
        "std1": std1,
        "mean2": mean2,
        "std2": std2,
    }


def _run_h1_batch(batch_start: int, batch_size: int, seed: int) -> tuple[int, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pre_change_sample = np.asarray(_WORKER_STATE["pre_change_sample"], dtype=np.float64)
    prepared_reference = _WORKER_STATE["prepared_reference"]
    sample_size = int(_WORKER_STATE["sample_size"])

    okcusum_stats = np.zeros((batch_size, sample_size), dtype=np.float64)
    scanb_stats = np.zeros((batch_size, sample_size), dtype=np.float64)

    for local_idx in range(batch_size):
        post_change_sample = sample_post_change_mixture(
            rng=rng,
            sample_size=sample_size,
            sample_dim=pre_change_sample.shape[1],
            mix_p=float(_WORKER_STATE["mix_p"]),
            mean1=float(_WORKER_STATE["mean1"]),
            std1=float(_WORKER_STATE["std1"]),
            mean2=float(_WORKER_STATE["mean2"]),
            std2=float(_WORKER_STATE["std2"]),
        )
        okcusum_stat, scanb_stat = online_kernel_cusum_statistic(
            pre_change_sample=pre_change_sample,
            post_change_sample=post_change_sample,
            omega_B=np.asarray(prepared_reference["omega_B"], dtype=int),
            num_blocks=int(prepared_reference["num_blocks"]),
            kernel_bandwidth=float(prepared_reference["kernel_bandwidth"]),
            prepared_reference=prepared_reference,
        )
        okcusum_stats[local_idx] = okcusum_stat
        scanb_stats[local_idx] = scanb_stat

    return batch_start, okcusum_stats, scanb_stats


def _H0_TASK_WRAPPER(task: tuple[int, int, int]) -> tuple[int, np.ndarray, np.ndarray]:
    return _run_h0_batch(*task)


def _H1_TASK_WRAPPER(task: tuple[int, int, int]) -> tuple[int, np.ndarray, np.ndarray]:
    return _run_h1_batch(*task)


def _run_batches_in_pool(
    tasks: list[tuple[int, int, int]],
    worker_fn,
    init_fn,
    init_args: tuple,
    desc: str,
    num_workers: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if num_workers <= 1:
        init_fn(*init_args)
        results = []
        for task in tqdm(tasks, desc=desc):
            results.append(worker_fn(*task))
        return results

    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=num_workers, initializer=init_fn, initargs=init_args) as pool:
        results = []
        for result in tqdm(
            pool.imap_unordered(
                _H0_TASK_WRAPPER if worker_fn is _run_h0_batch else _H1_TASK_WRAPPER,
                tasks,
            ),
            total=len(tasks),
            desc=desc,
        ):
            results.append(result)
    return results


def compute_h0_maxima(
    pre_change_sample: np.ndarray,
    prepared_reference: dict[str, np.ndarray | float | int],
    num_trials: int,
    calibration_horizon: int,
    seed: int,
    num_workers: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    okcusum_max = np.zeros(num_trials, dtype=np.float64)
    scanb_max = np.zeros(num_trials, dtype=np.float64)
    batch_size = max(1, chunk_size)
    num_batches = math.ceil(num_trials / batch_size)
    tasks = []
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        this_batch_size = min(batch_size, num_trials - batch_start)
        tasks.append((batch_start, this_batch_size, seed + batch_idx))

    results = _run_batches_in_pool(
        tasks=tasks,
        worker_fn=_run_h0_batch,
        init_fn=_init_h0_worker,
        init_args=(pre_change_sample, prepared_reference, calibration_horizon),
        desc="H0 calibration batches",
        num_workers=num_workers,
    )

    for batch_start, batch_okcusum, batch_scanb in results:
        batch_stop = batch_start + batch_okcusum.shape[0]
        okcusum_max[batch_start:batch_stop] = batch_okcusum
        scanb_max[batch_start:batch_stop] = batch_scanb

    return okcusum_max, scanb_max


def compute_h1_statistics(
    pre_change_sample: np.ndarray,
    prepared_reference: dict[str, np.ndarray | float | int],
    num_trials: int,
    sample_size: int,
    mix_p: float,
    mean1: float,
    std1: float,
    mean2: float,
    std2: float,
    seed: int,
    num_workers: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    okcusum_stats = np.zeros((num_trials, sample_size), dtype=np.float64)
    scanb_stats = np.zeros((num_trials, sample_size), dtype=np.float64)
    batch_size = max(1, chunk_size)
    num_batches = math.ceil(num_trials / batch_size)
    tasks = []
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        this_batch_size = min(batch_size, num_trials - batch_start)
        tasks.append((batch_start, this_batch_size, seed + batch_idx))

    results = _run_batches_in_pool(
        tasks=tasks,
        worker_fn=_run_h1_batch,
        init_fn=_init_h1_worker,
        init_args=(pre_change_sample, prepared_reference, sample_size, mix_p, mean1, std1, mean2, std2),
        desc="H1 EDD batches",
        num_workers=num_workers,
    )

    for batch_start, batch_okcusum, batch_scanb in results:
        batch_stop = batch_start + batch_okcusum.shape[0]
        okcusum_stats[batch_start:batch_stop] = batch_okcusum
        scanb_stats[batch_start:batch_stop] = batch_scanb

    return okcusum_stats, scanb_stats


def calibrate_thresholds(
    max_stats: np.ndarray,
    target_arl: np.ndarray,
    calibration_horizon: int,
) -> np.ndarray:
    lower_quantile = np.exp(-calibration_horizon / target_arl)
    return np.quantile(max_stats, lower_quantile)


def compute_edd_curve(statistics: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    edd = np.zeros_like(thresholds, dtype=np.float64)
    for idx, threshold in enumerate(thresholds):
        delays = np.array([detection_delay(stat_seq, threshold) for stat_seq in statistics], dtype=np.float64)
        edd[idx] = np.mean(delays)
    return edd


def plot_figure(
    target_arl: np.ndarray,
    okcusum_edd: np.ndarray,
    scanb_edd: np.ndarray,
    output_dir: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.log10(target_arl), okcusum_edd, "-o", linewidth=2.5, markersize=7, color=(204 / 255, 0, 0))
    ax.plot(np.log10(target_arl), scanb_edd, "-.o", linewidth=2.5, markersize=7, color=(255 / 255, 153 / 255, 51 / 255))
    ax.set_xlabel("log10(ARL)")
    ax.set_ylabel("EDD")
    ax.set_title(title)
    ax.grid(False)
    ax.legend(["Proposed", "Scan B"])
    fig.tight_layout()
    fig.savefig(output_dir / "figure7_leftmost.png", dpi=200)
    fig.savefig(output_dir / "figure7_leftmost.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_arl = np.array(args.target_arl, dtype=np.float64)
    pre_change_sample = load_reference_sample(
        num_reference=args.num_reference,
        sample_dim=args.sample_dim,
        seed=args.seed,
    )
    bandwidth = median_heuristic_bandwidth(pre_change_sample)
    omega_B = np.arange(2, args.window_size + 1, 2, dtype=int)
    prepared_reference = prepare_okcusum_reference(
        pre_change_sample=pre_change_sample,
        omega_B=omega_B,
        num_blocks=args.num_blocks,
        kernel_bandwidth=bandwidth,
    )

    okcusum_h0_max, scanb_h0_max = compute_h0_maxima(
        pre_change_sample=pre_change_sample,
        prepared_reference=prepared_reference,
        num_trials=args.num_h0_trials,
        calibration_horizon=args.calibration_horizon,
        seed=args.seed + 1,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
    okcusum_thresholds = calibrate_thresholds(okcusum_h0_max, target_arl, args.calibration_horizon)
    scanb_thresholds = calibrate_thresholds(scanb_h0_max, target_arl, args.calibration_horizon)

    okcusum_h1_stats, scanb_h1_stats = compute_h1_statistics(
        pre_change_sample=pre_change_sample,
        prepared_reference=prepared_reference,
        num_trials=args.num_h1_trials,
        sample_size=args.sample_size,
        mix_p=args.mix_p,
        mean1=args.mean1,
        std1=args.std1,
        mean2=args.mean2,
        std2=args.std2,
        seed=args.seed + 2,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
    okcusum_edd = compute_edd_curve(okcusum_h1_stats, okcusum_thresholds)
    scanb_edd = compute_edd_curve(scanb_h1_stats, scanb_thresholds)

    title = (
        "p = N(0, I20), q = "
        f"{args.mix_p:.1f}N({args.mean1:.1f}, {args.std1**2:.1f}) + "
        f"{1.0 - args.mix_p:.1f}N({args.mean2:.1f}, {args.std2**2:.1f})"
    )
    plot_figure(target_arl, okcusum_edd, scanb_edd, args.output_dir, title)

    np.savez(
        args.output_dir / "figure7_leftmost_results.npz",
        target_arl=target_arl,
        okcusum_h0_max=okcusum_h0_max,
        scanb_h0_max=scanb_h0_max,
        okcusum_thresholds=okcusum_thresholds,
        scanb_thresholds=scanb_thresholds,
        okcusum_edd=okcusum_edd,
        scanb_edd=scanb_edd,
        bandwidth=bandwidth,
        mix_p=args.mix_p,
        mean1=args.mean1,
        std1=args.std1,
        mean2=args.mean2,
        std2=args.std2,
        num_h0_trials=args.num_h0_trials,
        num_h1_trials=args.num_h1_trials,
        calibration_horizon=args.calibration_horizon,
        sample_size=args.sample_size,
        num_blocks=args.num_blocks,
        window_size=args.window_size,
    )

    print("Bandwidth:", bandwidth)
    print("num_workers:", args.num_workers)
    print("chunk_size:", args.chunk_size)
    print("target_arl:", target_arl)
    print("okcusum_thresholds:", okcusum_thresholds)
    print("scanb_thresholds:", scanb_thresholds)
    print("okcusum_edd:", okcusum_edd)
    print("scanb_edd:", scanb_edd)


if __name__ == "__main__":
    main()
