import numpy as np
from tqdm import tqdm
import os
from utils.data import augment_sequence_with_replacement


def get_uniform_ewma_statistics(pilot_x, online_y, l, headstart):
    max_t, dim = online_y.shape
    S = np.cov(np.transpose(pilot_x))
    
    inv_reg = 1e-6
    inv_S = np.linalg.inv(S+inv_reg*np.eye(dim))
    #l_diag = np.diag(np.repeat(l, dim))
    
    wt = np.zeros(max_t)
    dz = np.zeros((max_t+1, dim))
    dz[0] = headstart
    
    for i in range(max_t):
        yi = online_y[i, :]
        zi = l*yi + (1-l)*dz[i, :]
        sigma_zi = ((2-l) / (l*(1 - ( (1-l)**(2*(i+1)) )))) * inv_S
        t = np.dot(zi, np.matmul(zi, sigma_zi))
        
        wt[i] = t
        dz[i+1] = zi
        
    return wt, dz


def run_uniform_ewma(l: float, headstart: float,
                     f0_length: int, f1_length: int, 
                     f0_sequence, f1_sequence,
                     iter_num: int, save_dir: str):
    
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

        Wt_precp, dz_precp = get_uniform_ewma_statistics(x_1, y_precp, l, headstart)
        Wt_postcp, dz_postcp = get_uniform_ewma_statistics(x_2, y_postcp, l, headstart)
            
        stat_record[i,:] = np.concatenate([Wt_precp, Wt_postcp], axis=0)

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "ewma_iter{}_pre{}_post{}.npy".format(iter_num, f0_length, f1_length)), 
            stat_record)