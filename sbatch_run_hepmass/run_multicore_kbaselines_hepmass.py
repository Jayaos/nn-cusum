import argparse
import math
import multiprocessing as mp
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from baselines.kcusum import get_kcusum
from baselines.okcusum import get_okcusum
from baselines.scanb import get_scanb
from utils.data import augment_sequence_with_replacement


_WORKER_STATE = {}


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--save_dir", type=str, default="./results/")
    p.add_argument("--dataset_dir", type=str, default="./data/")
    p.add_argument("--f0_len", type=int, required=True)
    p.add_argument("--f1_len", type=int, required=True)
    p.add_argument("--iter_num", type=int, required=True)
    p.add_argument("--window_size", type=int, default=100)
    p.add_argument("--num_blocks", type=int, default=25)
    p.add_argument("--delta", type=float, default=1.0 / 50.0)
    p.add_argument("--random_seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--chunk_size", type=int, default=10)

    return p.parse_args()


def _init_worker(config):
    global _WORKER_STATE
    _WORKER_STATE = config


def _build_tasks(iter_num, chunk_size):
    batch_size = max(1, chunk_size)
    num_batches = math.ceil(iter_num / batch_size)
    tasks = []
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        this_batch_size = min(batch_size, iter_num - batch_start)
        tasks.append((batch_start, this_batch_size))
    return tasks


def _prepare_sequences(f0_sequence, f1_sequence, iter_num, f0_chunk_size, f1_chunk_size):
    required_f0_len = iter_num * f0_chunk_size
    required_f1_len = iter_num * f1_chunk_size

    if f0_sequence.shape[0] < required_f0_len:
        print("f0 sequence do not have enough data")
        print("current f0 sequence length: {}, required length: {}".format(f0_sequence.shape[0], required_f0_len))
        f0_sequence = augment_sequence_with_replacement(f0_sequence, required_f0_len)

    if f1_sequence.shape[0] < required_f1_len:
        print("f1 sequence do not have enough data")
        print("current f1 sequence length: {}, required length: {}".format(f1_sequence.shape[0], required_f1_len))
        f1_sequence = augment_sequence_with_replacement(f1_sequence, required_f1_len)

    return f0_sequence, f1_sequence


def _run_batch(task):
    batch_start, batch_size = task
    config = _WORKER_STATE

    f0_length = config["f0_len"]
    f1_length = config["f1_len"]
    f0_chunk_size = config["f0_chunk_size"]
    f1_chunk_size = config["f1_chunk_size"]
    entire_sequence_len = f0_length + f1_length

    batch_results = {
        "okcusum": np.zeros((batch_size, entire_sequence_len), dtype=np.float32),
        "kcusum": np.zeros((batch_size, entire_sequence_len), dtype=np.float32),
        "scanb": np.zeros((batch_size, entire_sequence_len), dtype=np.float32),
    }

    random_seed = config["random_seed"]
    for local_idx in range(batch_size):
        global_idx = batch_start + local_idx

        x_chunk = config["f0_sequence"][global_idx * f0_chunk_size:(global_idx + 1) * f0_chunk_size, :]
        y_chunk = config["f1_sequence"][global_idx * f1_chunk_size:(global_idx + 1) * f1_chunk_size, :]

        x_1 = np.float32(x_chunk[:f0_length])
        y_precp = np.float32(x_chunk[f0_length:2 * f0_length])
        x_2 = np.float32(x_chunk[2 * f0_length:])
        y_postcp = np.float32(y_chunk)

        batch_results["okcusum"][local_idx, :] = np.concatenate(
            [
                get_okcusum(
                    x_1,
                    y_precp,
                    config["num_blocks"],
                    config["window_size"],
                ),
                get_okcusum(
                    x_2,
                    y_postcp,
                    config["num_blocks"],
                    config["window_size"],
                ),
            ],
            axis=0,
        )

        if random_seed is not None:
            iter_rng = np.random.default_rng(random_seed + global_idx)
            rng_precp = np.random.default_rng(iter_rng.integers(0, 2**32 - 1))
            rng_postcp = np.random.default_rng(iter_rng.integers(0, 2**32 - 1))
        else:
            rng_precp = None
            rng_postcp = None

        batch_results["kcusum"][local_idx, :] = np.concatenate(
            [
                get_kcusum(
                    x_1,
                    y_precp,
                    delta=config["delta"],
                    rng=rng_precp,
                ),
                get_kcusum(
                    x_2,
                    y_postcp,
                    delta=config["delta"],
                    rng=rng_postcp,
                ),
            ],
            axis=0,
        )

        batch_results["scanb"][local_idx, :] = np.concatenate(
            [
                get_scanb(
                    x_1,
                    y_precp,
                    config["num_blocks"],
                    config["window_size"],
                ),
                get_scanb(
                    x_2,
                    y_postcp,
                    config["num_blocks"],
                    config["window_size"],
                ),
            ],
            axis=0,
        )

    return batch_start, batch_results


def _run_batches(tasks, config, num_workers):
    if num_workers <= 1:
        _init_worker(config)
        results = []
        for task in tqdm(tasks, desc="batches"):
            results.append(_run_batch(task))
        return results

    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(config,)) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(_run_batch, tasks), total=len(tasks), desc="batches"):
            results.append(result)
        return results


def _save_results(stat_record, args):
    print("saving results...")

    okcusum_save_dir = os.path.join(args.save_dir, "okcusum/")
    kcusum_save_dir = os.path.join(args.save_dir, "kcusum/")
    scanb_save_dir = os.path.join(args.save_dir, "scanb/")

    os.makedirs(okcusum_save_dir, exist_ok=True)
    os.makedirs(kcusum_save_dir, exist_ok=True)
    os.makedirs(scanb_save_dir, exist_ok=True)

    np.save(
        os.path.join(
            okcusum_save_dir,
            "okcusum_iter{}_pre{}_post{}_w{}_nb{}.npy".format(
                args.iter_num, args.f0_len, args.f1_len, args.window_size, args.num_blocks
            ),
        ),
        stat_record["okcusum"],
    )
    np.save(
        os.path.join(
            kcusum_save_dir,
            "kcusum_iter{}_pre{}_post{}_d{}.npy".format(
                args.iter_num, args.f0_len, args.f1_len, args.delta
            ),
        ),
        stat_record["kcusum"],
    )
    np.save(
        os.path.join(
            scanb_save_dir,
            "scanb_iter{}_pre{}_post{}_w{}_nb{}.npy".format(
                args.iter_num, args.f0_len, args.f1_len, args.window_size, args.num_blocks
            ),
        ),
        stat_record["scanb"],
    )


if __name__ == "__main__":
    print("python:", sys.version)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    args = parse_args()

    signal = np.load(os.path.join(args.dataset_dir, "signal_500.npy"))
    background = np.load(os.path.join(args.dataset_dir, "background.npy"))

    f0_chunk_size = 2 * args.f0_len + args.f1_len
    f1_chunk_size = args.f1_len
    entire_sequence_len = args.f0_len + args.f1_len

    background, signal = _prepare_sequences(
        f0_sequence=background,
        f1_sequence=signal,
        iter_num=args.iter_num,
        f0_chunk_size=f0_chunk_size,
        f1_chunk_size=f1_chunk_size,
    )

    tasks = _build_tasks(args.iter_num, args.chunk_size)
    config = {
        "f0_len": args.f0_len,
        "f1_len": args.f1_len,
        "f0_chunk_size": f0_chunk_size,
        "f1_chunk_size": f1_chunk_size,
        "window_size": args.window_size,
        "num_blocks": args.num_blocks,
        "delta": args.delta,
        "random_seed": args.random_seed,
        "f0_sequence": background,
        "f1_sequence": signal,
    }

    stat_record = {
        "okcusum": np.zeros((args.iter_num, entire_sequence_len), dtype=np.float32),
        "kcusum": np.zeros((args.iter_num, entire_sequence_len), dtype=np.float32),
        "scanb": np.zeros((args.iter_num, entire_sequence_len), dtype=np.float32),
    }

    results = _run_batches(tasks, config, args.num_workers)
    for batch_start, batch_results in results:
        batch_stop = batch_start + batch_results["okcusum"].shape[0]
        for method_name in stat_record:
            stat_record[method_name][batch_start:batch_stop] = batch_results[method_name]

    _save_results(stat_record, args)
