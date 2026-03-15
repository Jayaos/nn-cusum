"""
Contrastive online change point detection
https://proceedings.mlr.press/v206/puchkin23a.html

based on the author's code
https://github.com/npuchkin/contrastive_change_point_detection/tree/main
"""
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from model import MLP
from utils.data import augment_sequence_with_replacement


def get_cocpd_statistics(X, hidden_dims, stride, t_min=20, n_out_min=10, delta_max=50, 
                         n_epochs=1, learning_rate=0.01, device="cpu"):
    
    # X = X.reshape(-1, 1)
    # t_min: time t to start
    # n_out_min: gap between the current t and window
    # delta_max: is amount to the window size
    
    # Sample size
    n, dim_input = X.shape
    T = np.zeros((n, n))
    
    for t in tqdm(range(t_min, n, stride)):
        
        for tau in range(np.maximum(t - n_out_min - delta_max, n_out_min), t-n_out_min):
            
            # Initialize neural network
            f = MLP(dim_input, hidden_dims, 1).to(device)
            
            # Parameters of the optimizer
            opt = torch.optim.Adam(f.parameters(), lr=learning_rate)
            
            X_t = torch.tensor(X[:t, :], dtype=torch.float32, requires_grad=True)
            
            # weights
            W = torch.cat((torch.ones(tau) * (t - tau), torch.ones(t - tau) * tau)).reshape(-1, 1) # length t tensor
            
            # Create "virtual" labels
            Y_t = torch.cat((torch.ones(tau), torch.zeros(t - tau))).reshape(-1, 1) #  length t tensor
    
            # Loss function    
            loss_fn = nn.BCEWithLogitsLoss(weight=W)
            
            # Neural network training
            for epoch in range(n_epochs):
                
                loss = loss_fn(f(X_t), Y_t).mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
                
            Z = f(X_t).detach().numpy().reshape(-1)
            
            # Use thresholding to avoid numerical issues
            logit_clip = 10 # this is the threshold set by the original paper implementation (B in the original implementation)
            Z = np.minimum(Z, logit_clip)
            Z = np.maximum(Z, -logit_clip)
            
            D = np.zeros(t)
            D[:tau] = 2 / (1 + np.exp(-Z[:tau]))
            D[tau:] = 2 / (1 + np.exp(Z[tau:]))
            D = np.log(D)
            
            # Compute statistics for each t
            # and each change point candidate tau
            T[tau, t] = tau * (t - tau) / t * (np.mean(D[:tau]) + np.mean(D[tau:]))
       
    # Array of test statistics
    S = np.max(T, axis=0)
    
    return S, T


def run_cocpd(hidden_dims: list, window_size: int, stride: int, learning_rate: float,
              f0_length: int, f1_length: int, f0_sequence, f1_sequence, 
              iter_num: int, save_dir: str, device: str):
        
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

        sequence = np.concatenate([x_chunk, y_chunk])
        
        detection_statistics, _ = get_cocpd_statistics(sequence, dim_hidden=64, stride=stride, t_min=20, n_out_min=10, 
                                                          delta_max=window_size, n_epochs=epoch, 
                                                          learning_rate=learning_rate)
        
        stat_record[i,:] = detection_statistics
    
    saving = saving_dir + "nn-cocpd_stat_record_iter{}_p{}_o{}_s{}_w{}_e{}".format(iter_num, pilot_size, online_size,
                                                                                    stride, window_size, epoch)
    
    print("saving results...")
    np.save(saving, stat_record)