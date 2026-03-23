import numpy as np
import os
from tqdm import tqdm

from baselines._kernel_mmd import kcusum_statistic, median_heuristic_bandwidth
from utils.data import augment_sequence_with_replacement


def get_kcusum(pilot_X, arrival_Y, delta=1.0 / 50.0, kernel_bandwidth=None, rng=None):
    bandwidth = median_heuristic_bandwidth(pilot_X) if kernel_bandwidth is None else kernel_bandwidth
    return kcusum_statistic(
        pre_change_sample=pilot_X,
        post_change_sample=arrival_Y,
        kernel_bandwidth=bandwidth,
        delta=delta,
        rng=rng,
    )


def run_kcusum(f0_length: int, f1_length: int,
               f0_sequence, f1_sequence,
               iter_num: int, save_dir: str,
               delta: float = 1.0 / 50.0,
               kernel_bandwidth=None,
               random_seed: int | None = None):

    entire_sequence_len = f0_length + f1_length
    f0_chunk_size = 2 * f0_length + f1_length
    f1_chunk_size = f1_length

    stat_record = np.zeros(shape=(iter_num, entire_sequence_len))

    if f0_sequence.shape[0] < iter_num * f0_chunk_size:
        print("f0 sequence do not have enough data")
        print("current f0 sequence length: {}, required length: {}".format(f0_sequence.shape[0], iter_num * f0_chunk_size))
        f0_sequence = augment_sequence_with_replacement(f0_sequence, iter_num * f0_chunk_size)

    if f1_sequence.shape[0] < iter_num * f1_chunk_size:
        print("f1 sequence do not have enough data")
        print("current f1 sequence length: {}, required length: {}".format(f1_sequence.shape[0], iter_num * f1_chunk_size))
        f1_sequence = augment_sequence_with_replacement(f1_sequence, iter_num * f1_chunk_size)

    base_rng = np.random.default_rng(random_seed)

    for i in tqdm(range(iter_num)):
        x_chunk = f0_sequence[i * f0_chunk_size:(i + 1) * f0_chunk_size, :]
        y_chunk = f1_sequence[i * f1_chunk_size:(i + 1) * f1_chunk_size, :]

        x_1 = np.float32(x_chunk[:f0_length])
        y_precp = np.float32(x_chunk[f0_length:2 * f0_length])
        x_2 = np.float32(x_chunk[2 * f0_length:])
        y_postcp = np.float32(y_chunk)

        rng_precp = np.random.default_rng(base_rng.integers(0, 2**32 - 1)) if random_seed is not None else None
        rng_postcp = np.random.default_rng(base_rng.integers(0, 2**32 - 1)) if random_seed is not None else None

        Wt_precp = get_kcusum(x_1, y_precp, delta=delta, kernel_bandwidth=kernel_bandwidth, rng=rng_precp)
        Wt_postcp = get_kcusum(x_2, y_postcp, delta=delta, kernel_bandwidth=kernel_bandwidth, rng=rng_postcp)

        stat_record[i, :] = np.concatenate([Wt_precp, Wt_postcp], axis=0)

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(
        os.path.join(save_dir, "kcusum_iter{}_pre{}_post{}_d{}.npy".format(iter_num, f0_length, f1_length, delta)),
        stat_record,
    )
