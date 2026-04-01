from baselines.ewma import run_uniform_ewma
from baselines.hotelling_cusum import run_hotelling_cusum
from baselines.wlcusum import run_wlcusum
from baselines.wlglr import run_wlglr
from utils.simulation import generate_gaussian_mean_shift_p, generate_gaussian_mean_shift_q
import argparse
import torch
import sys
import os


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--save_dir", type=str, default="./results/")
    p.add_argument("--dataset_type", type=str, default="susy_all")
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

    f0_chunk_size = 2 * args.f0_len + args.f1_len
    f1_chunk_size = args.f1_len

    f0_sequence = generate_gaussian_mean_shift_p(
        args.data_dim,
        args.iter_num * f0_chunk_size,
        args.seed,
    )
    f1_sequence = generate_gaussian_mean_shift_q(
        args.data_dim,
        args.iter_num * f1_chunk_size,
        args.ms_delta,
        args.seed,
    )

    run_hotelling_cusum(args.f0_len, args.f1_len, 
                        f0_sequence, 
                        f1_sequence, 
                        args.iter_num,
                        os.path.join(args.save_dir, "hotelling_cusum/"))
    
    run_uniform_ewma(0.3, 0, 
                     args.f0_len, args.f1_len,
                     f0_sequence, f1_sequence, args.iter_num, 
                     os.path.join(args.save_dir, "mewma/"))


    run_wlcusum(100, args.f0_len, args.f1_len, 
                f0_sequence, f1_sequence, args.iter_num,
                os.path.join(args.save_dir, "wlcusum/"))
    
    run_wlglr(100, args.f0_len, args.f1_len, 
              f0_sequence, f1_sequence, args.iter_num,
              os.path.join(args.save_dir, "wlglr/"))
