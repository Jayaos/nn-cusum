from baselines.ewma import run_uniform_ewma
from baselines.hotelling_cusum import run_hotelling_cusum
from baselines.wlcusum import run_wlcusum
from baselines.wlglr import run_wlglr
from utils.simulation import generate_gamma
import argparse
import torch
import sys
import os


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--save_dir", type=str, default="./results/")
    p.add_argument("--data_dim", type=int, required=True)
    p.add_argument("--f0_shape", type=float, required=True)
    p.add_argument("--f0_scale", type=float, required=True)
    p.add_argument("--f1_shape", type=float, required=True)
    p.add_argument("--f1_scale", type=float, required=True)
    p.add_argument("--f1_loc_shift", type=float, required=True)
    p.add_argument("--window_size", type=int, default=100)
    p.add_argument("--f0_len", type=int)
    p.add_argument("--f1_len", type=int)
    p.add_argument("--iter_num", type=int)
    p.add_argument("--seed", type=int, default=2026)

    return p.parse_args()


if __name__ == "__main__":
    print("python:", sys.version)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    args = parse_args()

    f0_chunk_size = 2 * args.f0_len + args.f1_len
    f1_chunk_size = args.f1_len

    f0_sequence = generate_gamma(
        args.f0_shape,
        args.f0_scale,
        args.data_dim,
        args.iter_num * f0_chunk_size,
        location_shift=None,
        seed=args.seed,
    )
    
    f1_sequence = generate_gamma(
        args.f1_shape,
        args.f1_scale,
        args.data_dim,
        args.iter_num * f1_chunk_size,
        location_shift=args.f1_loc_shift,
        seed=args.seed + 1,
    )

    run_hotelling_cusum(
        args.f0_len,
        args.f1_len,
        f0_sequence,
        f1_sequence,
        args.iter_num,
        os.path.join(args.save_dir, "hotelling_cusum/"),
    )

    run_uniform_ewma(
        0.3,
        0,
        args.f0_len,
        args.f1_len,
        f0_sequence,
        f1_sequence,
        args.iter_num,
        os.path.join(args.save_dir, "mewma/"),
    )

    run_wlcusum(
        args.window_size,
        args.f0_len,
        args.f1_len,
        f0_sequence,
        f1_sequence,
        args.iter_num,
        os.path.join(args.save_dir, "wlcusum/"),
    )

    run_wlglr(
        args.window_size,
        args.f0_len,
        args.f1_len,
        f0_sequence,
        f1_sequence,
        args.iter_num,
        os.path.join(args.save_dir, "wlglr/"),
    )
