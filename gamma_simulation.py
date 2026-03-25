from utils.simulation import gen_Gamma, compute_arl_edd
from utils.utils import save_data
import numpy as np
import os
from tqdm import tqdm
from nn_cusum import *
from onnc import *
from onnr import *

"""
code to reproduce simulation experiment using Gaussian mean shift
"""


def run_nncusum_simulation(data_dim: int, hidden_dims: list, window_size: int, stride: int, 
                           batch_size: int, learning_rate: float, 
                           pilot_length: int, cp_location: int, postcp_length: int, burnin_length: int,
                           iter_num: int, save_dir: str, device: str):
    
    entire_sequence_len = burnin_length + cp_location + postcp_length
    stat_record = np.zeros(shape=(iter_num, entire_sequence_len))

    for i in tqdm(range(iter_num)):

        # generate data
        X = gen_Gamma(1, 1, data_dim, pilot_length) 
        Ynull = gen_Gamma(1, 1, data_dim, cp_location+burnin_length)
        Y = gen_Gamma(1, 0.8, data_dim, postcp_length)

        pilot_X = np.float32(X)
        arrival_Y = np.float32(np.concatenate((Ynull,Y), axis=0))

        idxt_nn, Wt_nn, dWt_nn, model, mXt_nn, mYt_nn = test_statistic(hidden_dims, pilot_X, arrival_Y, stride, 
                                                                       window_size, window_size, batch_size, 
                                                                       learning_rate, [burnin_length, burnin_length+cp_location],
                                                                       device=device)
    
        stat_record[i,idxt_nn] = Wt_nn

    arl_list = 10**(np.arange(2,5.6,0.1))
    arl, edd = compute_arl_edd(
        stat_record[:, burnin_length:],
        postcp_length + cp_location,
        cp_location,
        iter_num,
        arl_list,
    )
    
    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 
                         "nncusum_gamma_dim{}_iter{}_pre{}_post{}_b{}_h{}_l{}_s{}_w{}".format(data_dim, iter_num, 
                            cp_location, postcp_length, burnin_length,  
                            hidden_dims[0], len(hidden_dims), stride, window_size)), stat_record)
    save_data(
        os.path.join(
            save_dir, 
            "nncusum_gamma_arledd_dim{}_iter{}_pre{}_post{}_b{}_h{}_l{}_s{}_w{}.pkl".format(
            data_dim, iter_num, 
            cp_location, postcp_length, burnin_length,  
            hidden_dims[0], len(hidden_dims), stride, window_size)
            ),
        {"arl" : arl, "edd" : edd}
    )


def run_onnc_simulation(data_dim: int, hidden_dims: list, window_size: int, stride: int, learning_rate: float, epoch: int,
                        pilot_length: int, cp_location: int, postcp_length: int, burnin_length: int,
                        iter_num: int, save_dir: str, device: str):
    
    entire_sequence_len = burnin_length + cp_location + postcp_length
    stat_record = np.zeros(shape=(iter_num, entire_sequence_len))

    for i in tqdm(range(iter_num)):

        # generate data
        X = gen_Gamma(1, 1, data_dim, pilot_length) 
        Ynull = gen_Gamma(1, 1, data_dim, cp_location+burnin_length)
        Y = gen_Gamma(1, 0.8, data_dim, postcp_length)

        pilot_X = np.float32(X)
        arrival_Y = np.float32(np.concatenate((Ynull, Y), axis=0))

        Wt_onnc, loss_record = onnc_statistic(
            hidden_dims,
            pilot_X,
            arrival_Y,
            stride,
            window_size,
            learning_rate,
            epoch,
            device=device,
        )

        stat_record[i, :] = Wt_onnc

    arl_list = 10 ** (np.arange(2, 5.6, 0.1))
    arl, edd = compute_arl_edd(
        stat_record[:, burnin_length:],
        postcp_length + cp_location,
        cp_location,
        iter_num,
        arl_list,
    )

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(
        os.path.join(
            save_dir,
            "onnc_gamma_dim{}_iter{}_pre{}_post{}_b{}_h{}_l{}_s{}_w{}_e{}".format(
                data_dim,
                iter_num,
                cp_location,
                postcp_length,
                burnin_length,
                hidden_dims[0],
                len(hidden_dims),
                stride,
                window_size,
                epoch,
            ),
        ),
        stat_record,
    )

    save_data(
        os.path.join(
            save_dir, 
            "onnc_gamma_arledd_dim{}_iter{}_pre{}_post{}_b{}_h{}_l{}_s{}_w{}_e{}.pkl".format(
                data_dim,
                iter_num,
                cp_location,
                postcp_length,
                burnin_length,
                hidden_dims[0],
                len(hidden_dims),
                stride,
                window_size,
                epoch,
            ),
        ),
        {"arl" : arl, "edd" : edd}
    )


def run_onnr_simulation(data_dim: int, hidden_dims: list, window_size: int, alpha:float, stride: int, learning_rate: float, 
                        epoch:int, pilot_length: int, cp_location: int, postcp_length: int, burnin_length: int,
                        iter_num: int, save_dir: str, device: str):
    
    entire_sequence_len = burnin_length + cp_location + postcp_length
    stat_record = np.zeros(shape=(iter_num, entire_sequence_len))

    for i in tqdm(range(iter_num)):

        # generate data
        X = gen_Gamma(1, 1, data_dim, pilot_length) 
        Ynull = gen_Gamma(1, 1, data_dim, cp_location+burnin_length)
        Y = gen_Gamma(1, 0.8, data_dim, postcp_length)

        pilot_X = np.float32(X)
        arrival_Y = np.float32(np.concatenate((Ynull, Y), axis=0))

        Wt_onnr, loss_record = onnr_statistic(
            hidden_dims,
            pilot_X,
            arrival_Y,
            alpha,
            stride,
            window_size,
            learning_rate,
            epoch,
            device=device,
        )

        stat_record[i, :] = Wt_onnr

    arl_list = 10 ** (np.arange(2, 5.6, 0.1))
    arl, edd = compute_arl_edd(
        stat_record[:, burnin_length:],
        postcp_length + cp_location,
        cp_location,
        iter_num,
        arl_list,
    )

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(
        os.path.join(
            save_dir,
            "onnr_gamma_dim{}_iter{}_pre{}_post{}_b{}_h{}_l{}_a{}_s{}_w{}_e{}".format(
                data_dim,
                iter_num,
                cp_location,
                postcp_length,
                burnin_length,
                hidden_dims[0],
                len(hidden_dims),
                alpha,
                stride,
                window_size,
                epoch,
            ),
        ),
        stat_record,
    )

    save_data(
        os.path.join(
            save_dir, 
            "onnr_gamma_arledd_dim{}_iter{}_pre{}_post{}_b{}_h{}_l{}_s{}_w{}_e{}.pkl".format(
                data_dim,
                iter_num,
                cp_location,
                postcp_length,
                burnin_length,
                hidden_dims[0],
                len(hidden_dims),
                stride,
                window_size,
                epoch,
            ),
        ),
        {"arl" : arl, "edd" : edd}
    )
