from onnc import run_onnc_separate_prepost
from onnr import run_onnr_separate_prepost
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
    p.add_argument("--hidden_dims", type=int, nargs="+")
    p.add_argument("--window_size", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--stride", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--epoch", type=int)
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

    if args.dataset_type == "higgs_all":

        signal = np.load(os.path.join(args.dataset_dir, "higgs_all_signal.npy"))
        background = np.load(os.path.join(args.dataset_dir, "higgs_all_background.npy"))

    elif args.dataset_type == "higgs_low":

        signal = np.load(os.path.join(args.dataset_dir, "higgs_low_signal.npy"))
        background = np.load(os.path.join(args.dataset_dir, "higgs_low_background.npy"))

    run_onnc_separate_prepost(args.hidden_dims, args.window_size, args.stride, args.learning_rate, args.epoch,
             args.f0_len, args.f1_len, args.burnin_len,
             background, 
             signal, 
             args.iter_num,
             os.path.join(args.save_dir, "onnc/"),
             args.device)
    
    run_onnr_separate_prepost(args.hidden_dims, args.window_size, args.alpha, args.stride, args.learning_rate, args.epoch,
             args.f0_len, args.f1_len, args.burnin_len,
             background, 
             signal, 
             args.iter_num,
             os.path.join(args.save_dir, "onnr/"),
             args.device)