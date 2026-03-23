from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from baselines.kcusum import get_kcusum
from baselines.okcusum import get_okcusum
from baselines.scanb import get_scanb
from baselines._kernel_mmd import median_heuristic_bandwidth


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "table3_setting1_kernel"
_WORKER_STATE: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the KCUSUM / Scan-B / Online Kernel CUSUM rows for Table 3 Setting 1."
    )
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num_h0_trials", type=int, default=1000)
    parser.add_argument("--num_h1_trials", type=int, default=1000)
    parser.add_argument("--num_reference", type=int, default=2500)
    parser.add_argument("--sample_dim", type=int, default=20)
    parser.add_argument("--sequence_length", type=int, default=1000)
    parser.add_argument("--change_point", type=int, default=100)
    parser.add_argument("--target_arl", type=float, default=1000.0)
    parser.add_argument("--window_size", type=int, default=80)
    parser.add_argument("--num_blocks", type=int, default=30)
    parser.add_argument("--delta_kcusum", type=float, default=1.0 / 50.0)
    parser.add_argument("--mixture_shift_prob", type=float, default=7.0 / 8.0)
    parser.add_argument("--mixture_shift_mean", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=10)
    return parser.parse_args()


def sample_pre_change(rng: np.random.Generator, num_samples: int, sample_dim: int) -> np.ndarray:
    return rng.normal(loc=0.0, scale=1.0, size=(num_samples, sample_dim)).astype(np.float32)


def sample_setting1_post_change(
    rng: np.random.Generator,
    num_samples: int,
    sample_dim: int,
    shift_prob: float,
    shift_mean: float,
) -> np.ndarray:
    shifted = rng.normal(loc=shift_mean, scale=1.0, size=(num_samples, sample_dim))
    base = rng.normal(loc=0.0, scale=1.0, size=(num_samples, sample_dim))
    choose_shifted = rng.binomial(1, shift_prob, size=num_samples).astype(bool)
    samples = base
    samples[choose_shifted] = shifted[choose_shifted]
    return samples.astype(np.float32)


def compute_method_statistics(
    pilot_x: np.ndarray,
    arrival_y: np.ndarray,
    num_blocks: int,
    window_size: int,
    delta_kcusum: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    bandwidth = median_heuristic_bandwidth(pilot_x)
    return {
        "kcusum": get_kcusum(
            pilot_X=pilot_x,
            arrival_Y=arrival_y,
            delta=delta_kcusum,
            kernel_bandwidth=bandwidth,
            rng=rng,
        ),
        "scanb": get_scanb(
            pilot_X=pilot_x,
            arrival_Y=arrival_y,
            num_blocks=num_blocks,
            window_size=window_size,
            kernel_bandwidth=bandwidth,
        ),
        "okcusum": get_okcusum(
            pilot_X=pilot_x,
            arrival_Y=arrival_y,
            num_blocks=num_blocks,
            window_size=window_size,
            kernel_bandwidth=bandwidth,
        ),
    }


def _init_worker(config: dict[str, object]) -> None:
    global _WORKER_STATE
    _WORKER_STATE = config


def _run_h0_batch(task: tuple[int, int, int]) -> tuple[int, dict[str, np.ndarray]]:
    batch_start, batch_size, seed = task
    rng = np.random.default_rng(seed)
    config = _WORKER_STATE

    batch_results = {
        "kcusum": np.zeros(batch_size, dtype=np.float64),
        "scanb": np.zeros(batch_size, dtype=np.float64),
        "okcusum": np.zeros(batch_size, dtype=np.float64),
    }

    for local_idx in range(batch_size):
        pilot_x = sample_pre_change(rng, int(config["num_reference"]), int(config["sample_dim"]))
        arrival_y = sample_pre_change(rng, int(config["sequence_length"]), int(config["sample_dim"]))
        method_stats = compute_method_statistics(
            pilot_x=pilot_x,
            arrival_y=arrival_y,
            num_blocks=int(config["num_blocks"]),
            window_size=int(config["window_size"]),
            delta_kcusum=float(config["delta_kcusum"]),
            rng=rng,
        )
        for method_name, stat_seq in method_stats.items():
            batch_results[method_name][local_idx] = np.max(stat_seq)

    return batch_start, batch_results


def _run_h1_batch(task: tuple[int, int, int]) -> tuple[int, dict[str, np.ndarray]]:
    batch_start, batch_size, seed = task
    rng = np.random.default_rng(seed)
    config = _WORKER_STATE

    batch_results = {
        "kcusum": np.zeros((batch_size, int(config["sequence_length"])), dtype=np.float64),
        "scanb": np.zeros((batch_size, int(config["sequence_length"])), dtype=np.float64),
        "okcusum": np.zeros((batch_size, int(config["sequence_length"])), dtype=np.float64),
    }

    for local_idx in range(batch_size):
        pilot_x = sample_pre_change(rng, int(config["num_reference"]), int(config["sample_dim"]))
        pre_segment = sample_pre_change(rng, int(config["change_point"]), int(config["sample_dim"]))
        post_segment = sample_setting1_post_change(
            rng=rng,
            num_samples=int(config["sequence_length"]) - int(config["change_point"]),
            sample_dim=int(config["sample_dim"]),
            shift_prob=float(config["mixture_shift_prob"]),
            shift_mean=float(config["mixture_shift_mean"]),
        )
        arrival_y = np.concatenate([pre_segment, post_segment], axis=0).astype(np.float32)
        method_stats = compute_method_statistics(
            pilot_x=pilot_x,
            arrival_y=arrival_y,
            num_blocks=int(config["num_blocks"]),
            window_size=int(config["window_size"]),
            delta_kcusum=float(config["delta_kcusum"]),
            rng=rng,
        )
        for method_name, stat_seq in method_stats.items():
            batch_results[method_name][local_idx] = stat_seq

    return batch_start, batch_results


def _build_tasks(num_trials: int, chunk_size: int, seed: int) -> list[tuple[int, int, int]]:
    batch_size = max(1, chunk_size)
    num_batches = math.ceil(num_trials / batch_size)
    tasks = []
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        this_batch_size = min(batch_size, num_trials - batch_start)
        tasks.append((batch_start, this_batch_size, seed + batch_idx))
    return tasks


def _run_batches(tasks, worker_fn, config, desc, num_workers):
    if num_workers <= 1:
        _init_worker(config)
        results = []
        for task in tqdm(tasks, desc=desc):
            results.append(worker_fn(task))
        return results

    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(config,)) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(worker_fn, tasks), total=len(tasks), desc=desc):
            results.append(result)
    return results


def collect_h0_maxima(args: argparse.Namespace) -> dict[str, np.ndarray]:
    tasks = _build_tasks(args.num_h0_trials, args.chunk_size, args.seed + 1)
    config = {
        "num_reference": args.num_reference,
        "sample_dim": args.sample_dim,
        "sequence_length": args.sequence_length,
        "num_blocks": args.num_blocks,
        "window_size": args.window_size,
        "delta_kcusum": args.delta_kcusum,
    }
    h0_maxima = {
        "kcusum": np.zeros(args.num_h0_trials, dtype=np.float64),
        "scanb": np.zeros(args.num_h0_trials, dtype=np.float64),
        "okcusum": np.zeros(args.num_h0_trials, dtype=np.float64),
    }
    results = _run_batches(tasks, _run_h0_batch, config, "H0 calibration batches", args.num_workers)
    for batch_start, batch_results in results:
        batch_stop = batch_start + batch_results["kcusum"].shape[0]
        for method_name in h0_maxima:
            h0_maxima[method_name][batch_start:batch_stop] = batch_results[method_name]
    return h0_maxima


def collect_h1_statistics(args: argparse.Namespace) -> dict[str, np.ndarray]:
    tasks = _build_tasks(args.num_h1_trials, args.chunk_size, args.seed + 10001)
    config = {
        "num_reference": args.num_reference,
        "sample_dim": args.sample_dim,
        "sequence_length": args.sequence_length,
        "change_point": args.change_point,
        "num_blocks": args.num_blocks,
        "window_size": args.window_size,
        "delta_kcusum": args.delta_kcusum,
        "mixture_shift_prob": args.mixture_shift_prob,
        "mixture_shift_mean": args.mixture_shift_mean,
    }
    h1_statistics = {
        "kcusum": np.zeros((args.num_h1_trials, args.sequence_length), dtype=np.float64),
        "scanb": np.zeros((args.num_h1_trials, args.sequence_length), dtype=np.float64),
        "okcusum": np.zeros((args.num_h1_trials, args.sequence_length), dtype=np.float64),
    }
    results = _run_batches(tasks, _run_h1_batch, config, "H1 evaluation batches", args.num_workers)
    for batch_start, batch_results in results:
        batch_stop = batch_start + batch_results["kcusum"].shape[0]
        for method_name in h1_statistics:
            h1_statistics[method_name][batch_start:batch_stop] = batch_results[method_name]
    return h1_statistics


def calibrate_thresholds(h0_maxima: dict[str, np.ndarray], target_arl: float, horizon: int) -> dict[str, float]:
    quantile_level = float(np.exp(-horizon / target_arl))
    return {method_name: float(np.quantile(maxima, quantile_level)) for method_name, maxima in h0_maxima.items()}


def summarize_h1(statistics: np.ndarray, threshold: float, change_point: int) -> dict[str, float]:
    delays = []
    false_alarm = 0
    failure = 0

    for stat_seq in statistics:
        hits = np.flatnonzero(stat_seq > threshold)
        if hits.size == 0:
            failure += 1
            continue
        detect_time = int(hits[0] + 1)
        if detect_time <= change_point:
            false_alarm += 1
            continue
        delays.append(detect_time - change_point)

    success = len(delays)
    return {
        "edd_mean": float(np.mean(delays)) if delays else float("nan"),
        "edd_std": float(np.std(delays)) if delays else float("nan"),
        "success": success,
        "false_alarm": false_alarm,
        "failure": failure,
    }


def print_summary(summary: dict[str, dict[str, float]], thresholds: dict[str, float]) -> None:
    print("Target ARL:", 1000)
    for method_name in ["kcusum", "scanb", "okcusum"]:
        result = summary[method_name]
        print("")
        print(method_name)
        print("threshold:", thresholds[method_name])
        print("EDD (std.): {:.1f} ({:.1f})".format(result["edd_mean"], result["edd_std"]))
        print("Success: {}/1000".format(result["success"]))
        print("False Alarm: {}/1000".format(result["false_alarm"]))
        print("Failure: {}/1000".format(result["failure"]))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    h0_maxima = collect_h0_maxima(args)
    thresholds = calibrate_thresholds(h0_maxima, args.target_arl, args.sequence_length)
    h1_statistics = collect_h1_statistics(args)

    summary = {
        method_name: summarize_h1(h1_statistics[method_name], thresholds[method_name], args.change_point)
        for method_name in ["kcusum", "scanb", "okcusum"]
    }

    np.savez(
        args.output_dir / "table3_setting1_kernel_results.npz",
        h0_kcusum=h0_maxima["kcusum"],
        h0_scanb=h0_maxima["scanb"],
        h0_okcusum=h0_maxima["okcusum"],
        h1_kcusum=h1_statistics["kcusum"],
        h1_scanb=h1_statistics["scanb"],
        h1_okcusum=h1_statistics["okcusum"],
        threshold_kcusum=thresholds["kcusum"],
        threshold_scanb=thresholds["scanb"],
        threshold_okcusum=thresholds["okcusum"],
        target_arl=args.target_arl,
        change_point=args.change_point,
        num_reference=args.num_reference,
        sequence_length=args.sequence_length,
        sample_dim=args.sample_dim,
        num_blocks=args.num_blocks,
        window_size=args.window_size,
        delta_kcusum=args.delta_kcusum,
        mixture_shift_prob=args.mixture_shift_prob,
        mixture_shift_mean=args.mixture_shift_mean,
    )

    print_summary(summary, thresholds)


if __name__ == "__main__":
    main()
