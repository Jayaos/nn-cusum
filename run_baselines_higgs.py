from baselines.ewma import run_uniform_ewma
from baselines.hotelling_cusum import run_hotelling_cusum
from baselines.wlcusum import run_wlcusum
from baselines.wlglr import run_wlglr
import argparse
import torch
import numpy as np
import sys
import os


def parse_args():

    p = argparse.ArgumentParser()

    # data / problem
    p.add_argument("--save_dir", type=str, default="./results/")
    p.add_argument("--dataset_type", type=str, default="higgs_all")
    p.add_argument("--dataset_dir", type=str, default="./data/")
    p.add_argument("--f0_len", type=int)
    p.add_argument("--f1_len", type=int)
    p.add_argument("--iter_num", type=int)

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

    run_hotelling_cusum(args.f0_len, args.f1_len, 
                        background, 
                        signal, 
                        args.iter_num,
                        os.path.join(args.save_dir, "hotelling_cusum/"))
    
    run_uniform_ewma(0.3, 0, 
                     args.f0_len, args.f1_len,
                     background, signal, args.iter_num, 
                     os.path.join(args.save_dir, "mewma/"))


    run_wlcusum(100, args.f0_len, args.f1_len, 
                background, signal, args.iter_num,
                os.path.join(args.save_dir, "wlcusum/"))
    
    run_wlglr(100, args.f0_len, args.f1_len, 
              background, signal, args.iter_num,
              os.path.join(args.save_dir, "wlglr/"))
