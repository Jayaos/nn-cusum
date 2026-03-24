from meanshift_simulation import (
    run_nncusum_simulation,
    run_onnc_simulation,
    run_onnr_simulation,
)
import argparse
import torch
import sys
import os


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--save_dir", type=str, default="./results/")
    p.add_argument("--data_dim", type=int)
    p.add_argument("--hidden_dims", type=int, nargs="+")
    p.add_argument("--window_size", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--stride", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--epoch", type=int)
    p.add_argument("--pilot_len", type=int)
    p.add_argument("--cp_loc", type=int)
    p.add_argument("--postcp_len", type=int)
    p.add_argument("--burnin_len", type=int)
    p.add_argument("--iter_num", type=int)
    p.add_argument("--device", type=str, default="cpu")

    return p.parse_args()


if __name__ == "__main__":
    print("python:", sys.version)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    args = parse_args()

    run_nncusum_simulation(
        args.data_dim,
        args.hidden_dims,
        args.window_size,
        args.stride,
        args.batch_size,
        args.learning_rate,
        args.pilot_len,
        args.cp_loc,
        args.postcp_len,
        args.burnin_len,
        args.iter_num,
        os.path.join(args.save_dir, "nn_cusum/"),
        args.device,
    )

    run_onnc_simulation(
        args.data_dim,
        args.hidden_dims,
        args.window_size,
        args.stride,
        args.learning_rate,
        args.epoch,
        args.pilot_len,
        args.cp_loc,
        args.postcp_len,
        args.burnin_len,
        args.iter_num,
        os.path.join(args.save_dir, "onnc/"),
        args.device,
    )

    run_onnr_simulation(
        args.data_dim,
        args.hidden_dims,
        args.window_size,
        args.alpha,
        args.stride,
        args.learning_rate,
        args.epoch,
        args.pilot_len,
        args.cp_loc,
        args.postcp_len,
        args.burnin_len,
        args.iter_num,
        os.path.join(args.save_dir, "onnr/"),
        args.device,
    )
