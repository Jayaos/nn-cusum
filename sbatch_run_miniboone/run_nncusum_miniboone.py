from nn_cusum import run_nncusum
import argparse
import torch
import numpy as np
import sys
import os


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--save_dir", type=str, default="./results/")
    p.add_argument("--dataset_dir", type=str, default="./data/")
    p.add_argument("--hidden_dims", type=int, nargs="+")
    p.add_argument("--window_size", type=int)
    p.add_argument("--stride", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--f0_len", type=int)
    p.add_argument("--f1_len", type=int)
    p.add_argument("--burnin_len", type=int)
    p.add_argument("--iter_num", type=int)
    p.add_argument("--device", type=str, default="cpu")

    return p.parse_args()


if __name__ == "__main__":
    print("python:", sys.version)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    args = parse_args()

    signal = np.load(os.path.join(args.dataset_dir, "miniboone_signal.npy"))
    background = np.load(os.path.join(args.dataset_dir, "miniboone_background.npy"))

    run_nncusum(args.hidden_dims, args.window_size, args.stride, args.batch_size, args.learning_rate, 
                args.f0_len, args.f1_len, args.burnin_len,
                background, 
                signal, 
                args.iter_num,
                os.path.join(args.save_dir, "nn_cusum/"), 
                args.device)