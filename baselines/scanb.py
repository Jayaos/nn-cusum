import numpy as np
import os
from tqdm import tqdm

from baselines._kernel_mmd import median_heuristic_bandwidth, scanb_statistic
from utils.data import augment_sequence_with_replacement


def get_scanb(pilot_X, arrival_Y, num_blocks, window_size, kernel_bandwidth=None):
    bandwidth = median_heuristic_bandwidth(pilot_X) if kernel_bandwidth is None else kernel_bandwidth
    return scanb_statistic(
        pre_change_sample=pilot_X,
        post_change_sample=arrival_Y,
        block_size=window_size,
        num_blocks=num_blocks,
        kernel_bandwidth=bandwidth,
    )


def run_scanb(window_size: int, num_blocks: int,
              f0_length: int, f1_length: int,
              f0_sequence, f1_sequence,
              iter_num: int, save_dir: str,
              kernel_bandwidth=None):

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

    for i in tqdm(range(iter_num)):
        x_chunk = f0_sequence[i * f0_chunk_size:(i + 1) * f0_chunk_size, :]
        y_chunk = f1_sequence[i * f1_chunk_size:(i + 1) * f1_chunk_size, :]

        x_1 = np.float32(x_chunk[:f0_length])
        y_precp = np.float32(x_chunk[f0_length:2 * f0_length])
        x_2 = np.float32(x_chunk[2 * f0_length:])
        y_postcp = np.float32(y_chunk)

        Wt_precp = get_scanb(x_1, y_precp, num_blocks, window_size, kernel_bandwidth=kernel_bandwidth)
        Wt_postcp = get_scanb(x_2, y_postcp, num_blocks, window_size, kernel_bandwidth=kernel_bandwidth)

        stat_record[i, :] = np.concatenate([Wt_precp, Wt_postcp], axis=0)

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(
        os.path.join(save_dir, "scanb_iter{}_pre{}_post{}_w{}_nb{}.npy".format(iter_num, f0_length, f1_length, window_size, num_blocks)),
        stat_record,
    )
