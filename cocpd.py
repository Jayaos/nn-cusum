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


class NN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        
        super(NN, self).__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_out)
        )
    
    def forward(self, x):
        
        return self.net(x)
    
def nn_cocpd_test_statistic(X, dim_hidden, stride, t_min=20, n_out_min=10, delta_max=50, n_epochs=1, learning_rate=0.01):
    
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
            f = NN(n_in=dim_input, n_hidden=dim_hidden, n_out=1)
            
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

def get_nn_cocpd_stat_record_sequence_based(saving_dir, iter_num, pilot_size, online_size, pilot_sequence, online_sequence,
                                            stride, window_size, learning_rate, epoch):
    
    stat_record = np.zeros(shape=(iter_num, pilot_size+online_size))

    x_chunk_size = pilot_size
    y_chunk_size = online_size
    
    assert pilot_sequence.shape[0] >= iter_num * x_chunk_size, "x do not have enough data"
    assert online_sequence.shape[0] >= iter_num * y_chunk_size, "y do not have enough data"
    
    for i in tqdm(range(iter_num)):
        
        x_chunk = pilot_sequence[i*x_chunk_size:(i+1)*x_chunk_size,:]
        y_chunk = online_sequence[i*y_chunk_size:(i+1)*y_chunk_size,:]

        sequence = np.concatenate([x_chunk, y_chunk])
        
        detection_statistics, _ = nn_cocpd_test_statistic(sequence, dim_hidden=64, stride=stride, t_min=20, n_out_min=10, 
                                                          delta_max=window_size, n_epochs=epoch, 
                                                          learning_rate=learning_rate)
        
        stat_record[i,:] = detection_statistics
    
    saving = saving_dir + "nn-cocpd_stat_record_iter{}_p{}_o{}_s{}_w{}_e{}".format(iter_num, pilot_size, online_size,
                                                                                    stride, window_size, epoch)
    
    print("saving results...")
    np.save(saving, stat_record)