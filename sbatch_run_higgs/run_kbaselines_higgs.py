from baselines.kcusum import run_kcusum
from baselines.okcusum import run_okcusum
from baselines.scanb import run_scanb
import argparse
import torch
import numpy as np
import sys
import os


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--save_dir", type=str, default="./results/")
    p.add_argument("--dataset_type", type=str, default="higgs_all")
    p.add_argument("--dataset_dir", type=str, default="./data/")
    p.add_argument("--f0_len", type=int, required=True)
    p.add_argument("--f1_len", type=int, required=True)
    p.add_argument("--iter_num", type=int, required=True)
    p.add_argument("--window_size", type=int, default=100)
    p.add_argument("--num_blocks", type=int, default=25)
    p.add_argument("--delta", type=float, default=1.0 / 50.0)
    p.add_argument("--random_seed", type=int, default=None)

    return p.parse_args()


if __name__ == "__main__":
    print("python:", sys.version)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    args = parse_args()
    

    if args.dataset_type == "higgs_all":

        signal = np.load(os.path.join(args.dataset_dir, "higgs_all_signal.npy"))
        background = np.load(os.path.join(args.dataset_dir, "higgs_all_background.npy"))

    elif args.dataset_type == "higgs_low":

        signal = np.load(os.path.join(args.dataset_dir, "higgs_low_signal.npy"))
        background = np.load(os.path.join(args.dataset_dir, "higgs_low_background.npy"))

    run_okcusum(
        window_size=args.window_size,
        num_blocks=args.num_blocks,
        f0_length=args.f0_len,
        f1_length=args.f1_len,
        f0_sequence=background,
        f1_sequence=signal,
        iter_num=args.iter_num,
        save_dir=os.path.join(args.save_dir, "okcusum/"),
    )

    run_kcusum(
        f0_length=args.f0_len,
        f1_length=args.f1_len,
        f0_sequence=background,
        f1_sequence=signal,
        iter_num=args.iter_num,
        save_dir=os.path.join(args.save_dir, "kcusum/"),
        delta=args.delta,
        random_seed=args.random_seed,
    )

    run_scanb(
        window_size=args.window_size,
        num_blocks=args.num_blocks,
        f0_length=args.f0_len,
        f1_length=args.f1_len,
        f0_sequence=background,
        f1_sequence=signal,
        iter_num=args.iter_num,
        save_dir=os.path.join(args.save_dir, "scanb/"),
    )
