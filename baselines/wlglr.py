import numpy as np
import os
from tqdm import tqdm
from utils.data import augment_sequence_with_replacement


def get_wlglr(arrival_Y,w):

    maxT, dim = arrival_Y.shape
    Wt = np.zeros(maxT)
    
    init_i = int(w/2) # initialization of the first window location 

    for i in range(init_i, maxT):
        Yi = arrival_Y[i,:]
        now_w = min(w,i)
        wy = arrival_Y[(i-now_w):i,:]

        suffstat_y = np.cumsum(wy[::-1,],0)
        eta_w = np.linalg.norm(suffstat_y,axis=1)**2
        eta_w = eta_w/np.arange(1,now_w+1)
        Wt[i] = np.max(eta_w)

    return Wt


def run_wlglr(window_size: int,
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

        Wt_precp = get_wlglr(y_precp, window_size)
        Wt_postcp = get_wlglr(y_postcp, window_size)
            
        stat_record[i,:] = np.concatenate([Wt_precp, Wt_postcp], axis=0)


    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "wlglr_iter{}_pre{}_post{}.npy".format(iter_num, f0_length, f1_length)), 
            stat_record)