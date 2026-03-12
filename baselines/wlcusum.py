import numpy as np
import copy
import os
from tqdm import tqdm
from utils.data import augment_sequence_with_replacement


def compute_wlcusum_shift(x, dim, mup,invcov_p, logdetSigp, muq, covq):

    inv_reg = 1e-6
    
    invcov_q = np.linalg.inv(covq+ inv_reg*np.eye(dim))
    logdetSigq = np.log(max(np.linalg.det(covq),inv_reg))
    
    logp = -1/2*(x-mup)@invcov_p@(x-mup).T-1/2*logdetSigp
    logq = -1/2*(x-muq)@invcov_q@(x-muq).T-1/2*logdetSigq

    return logq-logp


def get_wlcusum(pilot_X, arrival_Y, w):

    # training window parameters
    maxT, dim = arrival_Y.shape    
    pilotT, dim = pilot_X.shape
    # compute online statistic
    dWt, Wt = np.zeros(maxT), np.zeros(maxT)
    Wprevr = 0
    
    init_i = int(w/2) # initialization of the first window location 

    mup  = np.mean(pilot_X,0) # mle
    covp = np.cov(pilot_X.T)*(pilotT-1)/pilotT # mle
    inv_reg = 1e-6
    invcov_p = np.linalg.inv(covp+inv_reg*np.eye(dim))
    logdetSigp = np.log(max(np.linalg.det(covp),inv_reg))
    
    for i in range(init_i, maxT):

        Yi = arrival_Y[i,:]
        now_w = min(w,i)
        wy = arrival_Y[(i-now_w):i,:]

        muq  = np.mean(wy,0) # mle
        covq = np.cov(wy.T)*(now_w-1) / now_w # mle # co-linearity

        eta = compute_wlcusum_shift(Yi, dim, mup, invcov_p, logdetSigp, muq, covq)
        dWt[i] = eta*1
        Wt[i] =  max(0,Wprevr+eta)
        Wprevr = copy.copy(Wt[i])

    return Wt


def run_wlcusum(window_size: int, 
                f0_length: int, f1_length: int, 
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
        y_precp = np.float32(x_chunk[f0_length:2*f0_length])
        x_2 = np.float32(x_chunk[2*f0_length:])
        y_postcp = np.float32(y_chunk)

        Wt_precp = get_wlcusum(x_1, y_precp, window_size)
        Wt_postcp = get_wlcusum(x_2, y_postcp, window_size)
            
        stat_record[i,:] = np.concatenate([Wt_precp,Wt_postcp], axis=0)

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "wlcusum_iter{}_pre{}_post{}.npy".format(iter_num, f0_length, f1_length)), 
            stat_record)