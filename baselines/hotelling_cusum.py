import random
import numpy as np
import copy
import os
from tqdm import tqdm
from utils.data import augment_sequence_with_replacement


def get_hotelling_cusum(pilot_X, arrival_Y, nX_te):

    maxT, dim = arrival_Y.shape
    nXpool = pilot_X.shape[0]
    nX_tr = nXpool-nX_te
    samplex_random_idx = random.sample(range(nXpool), nXpool) 
    X_tr = pilot_X[samplex_random_idx[:nX_tr],:] # train
    X_te = pilot_X[samplex_random_idx[nX_tr:nX_tr+nX_te] ,:] # test
    
    # estimate mean and cov of p from X_tr
    mean_p  = np.mean(X_tr,axis=0)
    S = np.cov( np.transpose(X_tr))
    inv_reg = 1e-6
    invcov_p = np.linalg.inv(S+ inv_reg*np.eye(dim))

    # estimate d:=E_{x~p}logq(x) using X_te
    eta_x = np.zeros(X_te.shape[0])
    for i in range(X_te.shape[0]):
        xi = X_te[i,:]
        eta_x[i]=1/2*np.dot( (xi-mean_p),  np.matmul(invcov_p,  (xi-mean_p)))
    #print(np.mean(eta_x), np.std(eta_x))
    d = np.mean(eta_x)*1.0
    # print(f'hotelling correction {d:>f}')

    # compute online statistic
    dWt, Wt = np.zeros(maxT), np.zeros(maxT)
    Wprevr = 0
    for i in range(maxT):
        Yi = arrival_Y[i,:]
        eta = 1/2*np.dot((Yi-mean_p), np.matmul(invcov_p, (Yi-mean_p)))-d
        dWt[i] = eta
        Wt[i] =  max(0, eta+Wprevr)
        Wprevr = copy.copy(Wt[i])

    return Wt, dWt


def run_hotelling_cusum(f0_length: int, f1_length: int, 
                        f0_sequence, f1_sequence,
                        iter_num: int, save_dir: str):
    
    # reference sequence: (f0_length + f1_length) construced from f0_sequence
    # online sequence: (f0_length) construced from f0_sequence, (f1_length) construced from f1_sequence
    entire_sequence_len = f0_length + f1_length
    f0_chunk_size = 2*f0_length+f1_length 
    f1_chunk_size = f1_length

    stat_record = np.zeros(shape=(iter_num, entire_sequence_len))

    if f0_sequence.shape[0] < iter_num*f0_chunk_size:
        print("f0 sequence do not have enough data")
        print("current f0 sequence length: {}, required length: {}".format(f0_sequence.shape[0], iter_num*f0_chunk_size))
        f0_sequence = augment_sequence_with_replacement(f0_sequence, iter_num*f0_chunk_size)

    if f1_sequence.shape[0] < iter_num * f1_chunk_size:
        print("f1 sequence do not have enough data")
        print("current f1 sequence length: {}, required length: {}".format(f1_sequence.shape[0], iter_num*f1_chunk_size))
        f1_sequence = augment_sequence_with_replacement(f1_sequence, iter_num*f1_chunk_size)
    
    for i in tqdm(range(iter_num)):
        
        x_chunk = f0_sequence[i*f0_chunk_size:(i+1)*f0_chunk_size,:]
        y_chunk = f1_sequence[i*f1_chunk_size:(i+1)*f1_chunk_size,:]

        x_1 = np.float32(x_chunk[:f0_length])
        y_pre_change = np.float32(x_chunk[f0_length:2*f0_length])
        x_2 = np.float32(x_chunk[2*f0_length:])
        y_post_change = np.float32(y_chunk)

        Wt_h_pre_change, dWt_precp = get_hotelling_cusum(x_1,y_pre_change, int(x_1.shape[0]/2))
        Wt_h_post_change, dWt_postcp = get_hotelling_cusum(x_2,y_post_change, int(x_2.shape[0]/2))
            
        stat_record[i,:] = np.concatenate([Wt_h_pre_change, Wt_h_post_change], axis=0)

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "hotelling_cusum_iter{}_pre{}_post{}.npy".format(iter_num, f0_length, f1_length)), 
            stat_record)