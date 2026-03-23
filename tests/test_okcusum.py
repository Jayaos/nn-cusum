import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from okcusum import median_heuristic_bandwidth, online_kernel_cusum_statistic


REPO_ROOT = Path(__file__).resolve().parents[1]
MATLAB_REFERENCE_CSV = REPO_ROOT / "online_kernel_cusum" / "raw_pre_change_sample_dim20.csv"

# By default, the mixture parameters mirror the MATLAB EDD-vs-ARL reproduction
# script in `online_kernel_cusum/example2_EDDvsARL.m`:
# q = 0.3 N(0, I) + 0.7 N(0, 4I).


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


def compute_h0_maxima(
    pre_change_sample: np.ndarray,
    num_trials: int,
    calibration_horizon: int,
    num_blocks: int,
    window_size: int,
    bandwidth: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    okcusum_max = np.zeros(num_trials, dtype=np.float64)
    scanb_max = np.zeros(num_trials, dtype=np.float64)
    omega_B = np.arange(2, window_size + 1, dtype=int)

    for trial_idx in tqdm(range(num_trials), desc="H0 calibration trials"):
        post_change_sample = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(calibration_horizon, pre_change_sample.shape[1]),
        )
        okcusum_stat, scanb_stat = online_kernel_cusum_statistic(
            pre_change_sample=pre_change_sample,
            post_change_sample=post_change_sample,
            omega_B=omega_B,
            num_blocks=num_blocks,
            kernel_bandwidth=bandwidth,
        )
        okcusum_max[trial_idx] = np.max(okcusum_stat)
        scanb_max[trial_idx] = np.max(scanb_stat)

    return okcusum_max, scanb_max


def compute_h1_statistics(
    pre_change_sample: np.ndarray,
    num_trials: int,
    sample_size: int,
    num_blocks: int,
    window_size: int,
    bandwidth: float,
    mix_p: float,
    mean1: float,
    std1: float,
    mean2: float,
    std2: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega_B = np.arange(2, window_size + 1, dtype=int)
    okcusum_stats = np.zeros((num_trials, sample_size), dtype=np.float64)
    scanb_stats = np.zeros((num_trials, sample_size), dtype=np.float64)

    for trial_idx in tqdm(range(num_trials), desc="H1 EDD trials"):
        post_change_sample = sample_post_change_mixture(
            rng=rng,
            sample_size=sample_size,
            sample_dim=pre_change_sample.shape[1],
            mix_p=mix_p,
            mean1=mean1,
            std1=std1,
            mean2=mean2,
            std2=std2,
        )
        okcusum_stat, scanb_stat = online_kernel_cusum_statistic(
            pre_change_sample=pre_change_sample,
            post_change_sample=post_change_sample,
            omega_B=omega_B,
            num_blocks=num_blocks,
            kernel_bandwidth=bandwidth,
        )
        okcusum_stats[trial_idx] = okcusum_stat
        scanb_stats[trial_idx] = scanb_stat

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

    okcusum_h0_max, scanb_h0_max = compute_h0_maxima(
        pre_change_sample=pre_change_sample,
        num_trials=args.num_h0_trials,
        calibration_horizon=args.calibration_horizon,
        num_blocks=args.num_blocks,
        window_size=args.window_size,
        bandwidth=bandwidth,
        seed=args.seed + 1,
    )
    okcusum_thresholds = calibrate_thresholds(okcusum_h0_max, target_arl, args.calibration_horizon)
    scanb_thresholds = calibrate_thresholds(scanb_h0_max, target_arl, args.calibration_horizon)

    okcusum_h1_stats, scanb_h1_stats = compute_h1_statistics(
        pre_change_sample=pre_change_sample,
        num_trials=args.num_h1_trials,
        sample_size=args.sample_size,
        num_blocks=args.num_blocks,
        window_size=args.window_size,
        bandwidth=bandwidth,
        mix_p=args.mix_p,
        mean1=args.mean1,
        std1=args.std1,
        mean2=args.mean2,
        std2=args.std2,
        seed=args.seed + 2,
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
    print("target_arl:", target_arl)
    print("okcusum_thresholds:", okcusum_thresholds)
    print("scanb_thresholds:", scanb_thresholds)
    print("okcusum_edd:", okcusum_edd)
    print("scanb_edd:", scanb_edd)


if __name__ == "__main__":
    main()
