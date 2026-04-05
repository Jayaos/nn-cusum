from baselines.kcusum import run_kcusum
from baselines.okcusum import run_okcusum
from baselines.scanb import run_scanb
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
    p.add_argument("--num_blocks", type=int, default=25)
    p.add_argument("--delta", type=float, default=1.0 / 50.0)
    p.add_argument("--f0_len", type=int, required=True)
    p.add_argument("--f1_len", type=int, required=True)
    p.add_argument("--iter_num", type=int, required=True)
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

    run_okcusum(
        window_size=args.window_size,
        num_blocks=args.num_blocks,
        f0_length=args.f0_len,
        f1_length=args.f1_len,
        f0_sequence=f0_sequence,
        f1_sequence=f1_sequence,
        iter_num=args.iter_num,
        save_dir=os.path.join(args.save_dir, "okcusum/"),
    )

    run_kcusum(
        f0_length=args.f0_len,
        f1_length=args.f1_len,
        f0_sequence=f0_sequence,
        f1_sequence=f1_sequence,
        iter_num=args.iter_num,
        save_dir=os.path.join(args.save_dir, "kcusum/"),
        delta=args.delta,
        random_seed=args.seed,
    )

    run_scanb(
        window_size=args.window_size,
        num_blocks=args.num_blocks,
        f0_length=args.f0_len,
        f1_length=args.f1_len,
        f0_sequence=f0_sequence,
        f1_sequence=f1_sequence,
        iter_num=args.iter_num,
        save_dir=os.path.join(args.save_dir, "scanb/"),
    )
