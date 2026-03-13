from onnc_onnr.online_change_rulsif import ChangePointDetectionOnline_RuLSIF
from tqdm import tqdm
import numpy as np
import os
from utils.data import augment_sequence_with_replacement


def run_onnr(hidden_dims: list, window_size: int, stride: int, lag_size: int,
                                            learning_rate: float, epoch: int,
                                            f0_length: int, f1_length: int, 
                                            f0_sequence, f1_sequence,
                                            iter_num: int, save_dir: str, device: str):
    
    # reference sequence: (f0_length + f1_length) construced from f0_sequence
    # online sequence: (f0_length) construced from f0_sequence, (f1_length) construced from f1_sequence
    entire_sequence_len = f0_length + f1_length
    shift_size = lag_size+2*window_size
    f0_chunk_size = f0_length+shift_size 
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
        
        reg = ChangePointDetectionOnline_RuLSIF(metric="None", alpha=0.1, periods=1, 
                                                batch_size=window_size, lag_size=lag_size, step=stride, n_epochs=epoch, 
                                                lr=learning_rate, lam=0.0, optimizer="Adam", device=device)
        
        sequence = np.concatenate([x_chunk, y_chunk])
        score_clf, peaks_clf = reg.predict(hidden_dims, sequence, height=1, smooth=False)   
        stat_record[i,:] = score_clf[:-shift_size]

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 
                         "onnr_iter{}_pre{}_post{}_s{}_w{}_e{}".format(iter_num, f0_length, f1_length, 
                                                                       stride, window_size, epoch)),
                                                                       stat_record)
    
