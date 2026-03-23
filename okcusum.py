from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from tqdm import tqdm

from utils.data import augment_sequence_with_replacement


def eu_dist2(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = x if y is None else np.asarray(y, dtype=np.float64)
    x_norm = np.sum(x * x, axis=1)[:, None]
    y_norm = np.sum(y * y, axis=1)[None, :]
    dist2 = x_norm + y_norm - 2.0 * x @ y.T
    return np.maximum(dist2, 0.0)


def rbf_kernel_from_dist2(dist2: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-dist2 / (2.0 * bandwidth**2))


def median_heuristic_bandwidth(samples: np.ndarray) -> float:
    dist2 = eu_dist2(samples)
    positive_dist2 = dist2[dist2 > 0]
    if positive_dist2.size == 0:
        raise ValueError("median heuristic requires at least two distinct samples")
    return float(np.median(positive_dist2))


def h_rbf(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray, bandwidth: float) -> float:
    d_xx = np.sum((x1 - x2) ** 2)
    d_yy = np.sum((y1 - y2) ** 2)
    d_xy = np.sum((x1 - y2) ** 2)
    d_yx = np.sum((x2 - y1) ** 2)
    return float(
        np.exp(-d_xx / (2.0 * bandwidth**2))
        + np.exp(-d_yy / (2.0 * bandwidth**2))
        - np.exp(-d_xy / (2.0 * bandwidth**2))
        - np.exp(-d_yx / (2.0 * bandwidth**2))
    )


def eh_square(pre_change_sample: np.ndarray, bandwidth: float) -> float:
    sz_ref = pre_change_sample.shape[0]
    sample_size = sz_ref // 4
    if sample_size == 0:
        raise ValueError("pre_change_sample is too small to estimate Eh_square")

    x = pre_change_sample[0:sample_size]
    x_p = pre_change_sample[sample_size : 2 * sample_size]
    y = pre_change_sample[2 * sample_size : 3 * sample_size]
    y_p = pre_change_sample[3 * sample_size : 4 * sample_size]

    d1 = np.diag(eu_dist2(x, x_p))
    d2 = np.diag(eu_dist2(y, y_p))
    d3 = np.diag(eu_dist2(x, y_p))
    d4 = np.diag(eu_dist2(x_p, y))

    h_sq = (
        np.exp(-d1 / (2.0 * bandwidth**2))
        + np.exp(-d2 / (2.0 * bandwidth**2))
        - np.exp(-d3 / (2.0 * bandwidth**2))
        - np.exp(-d4 / (2.0 * bandwidth**2))
    ) ** 2
    return float(np.mean(h_sq))


def covariance_h(pre_change_sample: np.ndarray, bandwidth: float) -> float:
    sz_ref = pre_change_sample.shape[0]
    sample_size = sz_ref // 6
    if sample_size == 0:
        raise ValueError("pre_change_sample is too small to estimate Covariance_h")

    x = pre_change_sample[0:sample_size]
    x_p = pre_change_sample[sample_size : 2 * sample_size]
    y = pre_change_sample[2 * sample_size : 3 * sample_size]
    y_p = pre_change_sample[3 * sample_size : 4 * sample_size]
    x_pp = pre_change_sample[4 * sample_size : 5 * sample_size]
    x_ppp = pre_change_sample[5 * sample_size : 6 * sample_size]

    d1 = np.diag(eu_dist2(x, x_p))
    d2 = np.diag(eu_dist2(y, y_p))
    d3 = np.diag(eu_dist2(x, y_p))
    d4 = np.diag(eu_dist2(x_p, y))
    d1_p = np.diag(eu_dist2(x_pp, x_ppp))
    d2_p = np.diag(eu_dist2(y, y_p))
    d3_p = np.diag(eu_dist2(x_pp, y_p))
    d4_p = np.diag(eu_dist2(x_ppp, y))

    k1 = (
        np.exp(-d1 / (2.0 * bandwidth**2))
        + np.exp(-d2 / (2.0 * bandwidth**2))
        - np.exp(-d3 / (2.0 * bandwidth**2))
        - np.exp(-d4 / (2.0 * bandwidth**2))
    )
    k2 = (
        np.exp(-d1_p / (2.0 * bandwidth**2))
        + np.exp(-d2_p / (2.0 * bandwidth**2))
        - np.exp(-d3_p / (2.0 * bandwidth**2))
        - np.exp(-d4_p / (2.0 * bandwidth**2))
    )
    return float(np.mean(k1 * k2) - np.mean(k1) * np.mean(k2))


def _normalize_omega_B(omega_B: Iterable[int]) -> np.ndarray:
    omega_B = np.array(sorted(set(int(b) for b in omega_B)), dtype=int)
    if omega_B.size == 0 or omega_B[0] < 2:
        raise ValueError("omega_B must contain integers greater than or equal to 2")
    return omega_B


def _scanb_variance_estimates(pre_change_sample: np.ndarray, omega_B: np.ndarray, num_blocks: int, bandwidth: float) -> np.ndarray:
    eh_sq = eh_square(pre_change_sample, bandwidth)
    cov_h = covariance_h(pre_change_sample, bandwidth)
    numerator = eh_sq / num_blocks + (1.0 - 1.0 / num_blocks) * cov_h
    return np.array([numerator / (b * (b - 1) / 2.0) for b in omega_B], dtype=np.float64)


def online_kernel_cusum_statistic(
    pre_change_sample: np.ndarray,
    post_change_sample: np.ndarray,
    omega_B: Iterable[int],
    num_blocks: int,
    kernel_bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    omega_B = _normalize_omega_B(omega_B)
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    b_max = int(np.max(omega_B))
    pre_change_sample = np.asarray(pre_change_sample, dtype=np.float64)
    post_change_sample = np.asarray(post_change_sample, dtype=np.float64)

    required_ref = (num_blocks + 1) * b_max - 1
    if pre_change_sample.shape[0] < required_ref:
        raise ValueError(
            f"pre_change_sample must have at least {required_ref} rows for num_blocks={num_blocks} and max block={b_max}"
        )

    variance_est = _scanb_variance_estimates(pre_change_sample, omega_B, num_blocks, kernel_bandwidth)

    all_sample = np.concatenate([pre_change_sample, post_change_sample], axis=0)
    sample_size_ref = pre_change_sample.shape[0]
    sample_size = post_change_sample.shape[0]

    ref_start = sample_size_ref - (num_blocks + 1) * b_max + 1
    ref_end = sample_size_ref - b_max + 1
    reference_data = all_sample[ref_start:ref_end]

    detect_stat_seq_max = np.zeros(sample_size, dtype=np.float64)
    detect_stat_seq_sw = np.zeros(sample_size, dtype=np.float64)

    dxx = np.zeros((b_max, b_max), dtype=np.float64)
    dyy_collection = np.zeros((num_blocks, b_max, b_max), dtype=np.float64)
    dxy_collection = np.zeros((num_blocks, b_max, b_max), dtype=np.float64)

    for blk_idx in range(num_blocks):
        start = blk_idx * b_max
        stop = (blk_idx + 1) * b_max
        pre_change_block = reference_data[start:stop]
        dyy_collection[blk_idx] = eu_dist2(pre_change_block)

    for t_idx in range(sample_size):
        absolute_t = sample_size_ref + t_idx
        post_change_block = all_sample[absolute_t - b_max + 1 : absolute_t + 1]

        if t_idx == 0:
            dxx = eu_dist2(post_change_block)
            for blk_idx in range(num_blocks):
                start = blk_idx * b_max
                stop = (blk_idx + 1) * b_max
                pre_change_block = reference_data[start:stop]
                dxy_collection[blk_idx] = eu_dist2(post_change_block, pre_change_block)
        else:
            new_sample = post_change_block[-1:]
            past_post_sample = post_change_block[:-1]

            reused_dxx = dxx[1:, 1:]
            new_dxx_off_diag = eu_dist2(new_sample, past_post_sample)
            new_dxx_diag = eu_dist2(new_sample, new_sample)

            dxx[:-1, :-1] = reused_dxx
            dxx[-1:, -1:] = new_dxx_diag
            dxx[-1:, :-1] = new_dxx_off_diag
            dxx[:-1, -1:] = new_dxx_off_diag.T

            for blk_idx in range(num_blocks):
                start = blk_idx * b_max
                stop = (blk_idx + 1) * b_max
                pre_change_block = reference_data[start:stop]
                reused_dxy = dxy_collection[blk_idx, 1:, :]
                new_dxy = eu_dist2(new_sample, pre_change_block)
                dxy_collection[blk_idx, :-1, :] = reused_dxy
                dxy_collection[blk_idx, -1:, :] = new_dxy

        max_kernel_cpd_stat = -np.inf
        scan_b_stat = 0.0

        for block_idx, block_size in enumerate(omega_B):
            start = b_max - block_size
            temp1 = rbf_kernel_from_dist2(dxx[start:, start:], kernel_bandwidth)

            temp_stat = 0.0
            for blk_idx in range(num_blocks):
                dxy = dxy_collection[blk_idx, start:, start:]
                dyy = dyy_collection[blk_idx, start:, start:]
                temp2 = rbf_kernel_from_dist2(dyy, kernel_bandwidth)
                temp3 = rbf_kernel_from_dist2(dxy, kernel_bandwidth)
                temp = temp1 + temp2 - temp3 - temp3.T
                mmd2 = (np.sum(temp) - np.trace(temp)) / (block_size * (block_size - 1))
                temp_stat += mmd2 / num_blocks

            temp_stat /= np.sqrt(variance_est[block_idx])
            if temp_stat > max_kernel_cpd_stat:
                max_kernel_cpd_stat = temp_stat
            if block_size == b_max:
                scan_b_stat = temp_stat

        detect_stat_seq_max[t_idx] = max_kernel_cpd_stat
        detect_stat_seq_sw[t_idx] = scan_b_stat

    return detect_stat_seq_max, detect_stat_seq_sw


def scanb_statistic(
    pre_change_sample: np.ndarray,
    post_change_sample: np.ndarray,
    block_size: int,
    num_blocks: int,
    kernel_bandwidth: float,
) -> np.ndarray:
    _, scan_b = online_kernel_cusum_statistic(
        pre_change_sample=pre_change_sample,
        post_change_sample=post_change_sample,
        omega_B=[block_size],
        num_blocks=num_blocks,
        kernel_bandwidth=kernel_bandwidth,
    )
    return scan_b


def kcusum_statistic(
    pre_change_sample: np.ndarray,
    post_change_sample: np.ndarray,
    kernel_bandwidth: float,
    delta: float = 1.0 / 50.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    pre_change_sample = np.asarray(pre_change_sample, dtype=np.float64)
    post_change_sample = np.asarray(post_change_sample, dtype=np.float64)
    sample_size_ref = pre_change_sample.shape[0]
    sample_size = post_change_sample.shape[0]

    rng = np.random.default_rng() if rng is None else rng
    detect_stat = np.zeros(sample_size, dtype=np.float64)

    for t_idx in range(1, sample_size, 2):
        post_change_block = post_change_sample[t_idx - 1 : t_idx + 1]
        pre_change_block = pre_change_sample[rng.permutation(sample_size_ref)[:2]]

        x0, x1 = pre_change_block
        y0, y1 = post_change_block
        increment = h_rbf(x0, x1, y0, y1, kernel_bandwidth) - delta
        detect_stat[t_idx] = max(0.0, detect_stat[t_idx - 1] + increment)

    if sample_size % 2 == 0:
        detect_stat[2 : sample_size - 1 : 2] = detect_stat[1 : sample_size - 2 : 2]
    else:
        detect_stat[2 : sample_size - 2 : 2] = detect_stat[1 : sample_size - 3 : 2]

    return detect_stat


def _prepare_run_inputs(
    f0_length: int,
    f1_length: int,
    burnin_length: int,
    f0_sequence: np.ndarray,
    f1_sequence: np.ndarray,
    iter_num: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    entire_sequence_len = f0_length + f1_length
    f0_with_burnin_length = f0_length + burnin_length
    f0_chunk_size = 2 * f0_with_burnin_length + f1_length
    f1_chunk_size = f1_length

    if f0_sequence.shape[0] < iter_num * f0_chunk_size:
        print("f0 sequence do not have enough data")
        print(f"current f0 sequence length: {f0_sequence.shape[0]}, required length: {iter_num * f0_chunk_size}")
        f0_sequence = augment_sequence_with_replacement(f0_sequence, iter_num * f0_chunk_size)

    if f1_sequence.shape[0] < iter_num * f1_chunk_size:
        print("f1 sequence do not have enough data")
        print(f"current f1 sequence length: {f1_sequence.shape[0]}, required length: {iter_num * f1_chunk_size}")
        f1_sequence = augment_sequence_with_replacement(f1_sequence, iter_num * f1_chunk_size)

    return f0_sequence, f1_sequence, entire_sequence_len, f0_with_burnin_length


def _build_iteration_sequences(
    f0_sequence: np.ndarray,
    f1_sequence: np.ndarray,
    iter_idx: int,
    f0_length: int,
    f1_length: int,
    burnin_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    f0_with_burnin_length = f0_length + burnin_length
    f0_chunk_size = 2 * f0_with_burnin_length + f1_length
    f1_chunk_size = f1_length

    x_chunk = f0_sequence[iter_idx * f0_chunk_size : (iter_idx + 1) * f0_chunk_size]
    y_chunk = f1_sequence[iter_idx * f1_chunk_size : (iter_idx + 1) * f1_chunk_size]

    arrival_y = np.float32(np.concatenate([x_chunk[:f0_with_burnin_length], y_chunk], axis=0))
    pilot_x = np.float32(x_chunk[f0_with_burnin_length:])
    return pilot_x, arrival_y


def run_okcusum(
    window_size: int,
    num_blocks: int,
    f0_length: int,
    f1_length: int,
    burnin_length: int,
    f0_sequence: np.ndarray,
    f1_sequence: np.ndarray,
    iter_num: int,
    save_dir: str,
    omega_B: Iterable[int] | None = None,
    kernel_bandwidth: float | None = None,
) -> np.ndarray:
    omega_B = np.arange(2, window_size + 1) if omega_B is None else _normalize_omega_B(omega_B)
    stat_record = np.zeros((iter_num, f0_length + f1_length), dtype=np.float64)
    f0_sequence, f1_sequence, _, _ = _prepare_run_inputs(
        f0_length=f0_length,
        f1_length=f1_length,
        burnin_length=burnin_length,
        f0_sequence=f0_sequence,
        f1_sequence=f1_sequence,
        iter_num=iter_num,
    )

    for i in tqdm(range(iter_num)):
        pilot_x, arrival_y = _build_iteration_sequences(
            f0_sequence=f0_sequence,
            f1_sequence=f1_sequence,
            iter_idx=i,
            f0_length=f0_length,
            f1_length=f1_length,
            burnin_length=burnin_length,
        )
        bandwidth = median_heuristic_bandwidth(pilot_x) if kernel_bandwidth is None else kernel_bandwidth
        okcusum_stat, _ = online_kernel_cusum_statistic(pilot_x, arrival_y, omega_B, num_blocks, bandwidth)
        stat_record[i] = okcusum_stat[burnin_length:]

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    filename = (
        f"okcusum_iter{iter_num}_pre{f0_length}_post{f1_length}_b{burnin_length}"
        f"_w{window_size}_nb{num_blocks}.npy"
    )
    np.save(os.path.join(save_dir, filename), stat_record)
    return stat_record


def run_scanb(
    window_size: int,
    num_blocks: int,
    f0_length: int,
    f1_length: int,
    burnin_length: int,
    f0_sequence: np.ndarray,
    f1_sequence: np.ndarray,
    iter_num: int,
    save_dir: str,
    kernel_bandwidth: float | None = None,
) -> np.ndarray:
    stat_record = np.zeros((iter_num, f0_length + f1_length), dtype=np.float64)
    f0_sequence, f1_sequence, _, _ = _prepare_run_inputs(
        f0_length=f0_length,
        f1_length=f1_length,
        burnin_length=burnin_length,
        f0_sequence=f0_sequence,
        f1_sequence=f1_sequence,
        iter_num=iter_num,
    )

    for i in tqdm(range(iter_num)):
        pilot_x, arrival_y = _build_iteration_sequences(
            f0_sequence=f0_sequence,
            f1_sequence=f1_sequence,
            iter_idx=i,
            f0_length=f0_length,
            f1_length=f1_length,
            burnin_length=burnin_length,
        )
        bandwidth = median_heuristic_bandwidth(pilot_x) if kernel_bandwidth is None else kernel_bandwidth
        scanb_stat = scanb_statistic(pilot_x, arrival_y, window_size, num_blocks, bandwidth)
        stat_record[i] = scanb_stat[burnin_length:]

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    filename = (
        f"scanb_iter{iter_num}_pre{f0_length}_post{f1_length}_b{burnin_length}"
        f"_w{window_size}_nb{num_blocks}.npy"
    )
    np.save(os.path.join(save_dir, filename), stat_record)
    return stat_record


def run_kcusum(
    f0_length: int,
    f1_length: int,
    burnin_length: int,
    f0_sequence: np.ndarray,
    f1_sequence: np.ndarray,
    iter_num: int,
    save_dir: str,
    kernel_bandwidth: float | None = None,
    delta: float = 1.0 / 50.0,
) -> np.ndarray:
    stat_record = np.zeros((iter_num, f0_length + f1_length), dtype=np.float64)
    f0_sequence, f1_sequence, _, _ = _prepare_run_inputs(
        f0_length=f0_length,
        f1_length=f1_length,
        burnin_length=burnin_length,
        f0_sequence=f0_sequence,
        f1_sequence=f1_sequence,
        iter_num=iter_num,
    )

    for i in tqdm(range(iter_num)):
        pilot_x, arrival_y = _build_iteration_sequences(
            f0_sequence=f0_sequence,
            f1_sequence=f1_sequence,
            iter_idx=i,
            f0_length=f0_length,
            f1_length=f1_length,
            burnin_length=burnin_length,
        )
        bandwidth = median_heuristic_bandwidth(pilot_x) if kernel_bandwidth is None else kernel_bandwidth
        kcusum_stat = kcusum_statistic(pilot_x, arrival_y, bandwidth, delta=delta)
        stat_record[i] = kcusum_stat[burnin_length:]

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    filename = (
        f"kcusum_iter{iter_num}_pre{f0_length}_post{f1_length}_b{burnin_length}"
        f"_d{delta}.npy"
    )
    np.save(os.path.join(save_dir, filename), stat_record)
    return stat_record
