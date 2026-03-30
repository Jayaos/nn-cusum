from nn_cusum import run_nncusum_separate_prepost
from utils.simulation import generate_gaussian_mean_shift_p, generate_gaussian_mean_shift_q
import argparse
import torch
import sys
import os


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--save_dir", type=str, default="./results/")
    p.add_argument("--data_dim", type=int)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--hidden_dims", type=int, nargs="+")
    p.add_argument("--window_size", type=int)
    p.add_argument("--stride", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--f0_len", type=int)
    p.add_argument("--f1_len", type=int)
    p.add_argument("--burnin_len", type=int)
    p.add_argument("--iter_num", type=int)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", type=str, default="cpu")

    return p.parse_args()


if __name__ == "__main__":
    print("python:", sys.version)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    args = parse_args()

    f0_with_burnin_length = args.f0_len + args.burnin_len
    f0_chunk_size = 2 * f0_with_burnin_length + args.f1_len
    f1_chunk_size = args.f1_len

    f0_sequence = generate_gaussian_mean_shift_p(args.data_dim, 
                                                 args.iter_num * f0_chunk_size,
                                                 args.seed)
    f1_sequence = generate_gaussian_mean_shift_q(args.data_dim, 
                                                 args.iter_num * f1_chunk_size, 
                                                 args.delta,
                                                 args.seed)

    run_nncusum_separate_prepost(
        args.hidden_dims,
        args.window_size,
        args.stride,
        args.batch_size,
        args.learning_rate,
        args.f0_len,
        args.f1_len,
        args.burnin_len,
        f0_sequence,
        f1_sequence,
        args.iter_num,
        os.path.join(args.save_dir, "nn_cusum/"),
        args.device,
    )
