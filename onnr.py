import numpy as np
import random
import torch
from tqdm import tqdm
from model import MLP
from utils.data import augment_sequence_with_replacement
import os
import math


def onnr_loss(output_1, output_2, alpha):

    return output_1.pow(2).mean() * ((1-alpha)/2) + output_2.pow(2).mean() * (alpha/2) - output_2.mean()

    
def train_onnr_batch(batch_x, batch_y, model, optimizer, epoch, alpha):

    assert batch_x.shape == batch_y.shape, print("batch_x and batch_y must have the same shape")

    model.train()
    loss_sum = 0.

    for e in range(epoch):
        optimizer.zero_grad()
        output_x = model(batch_x)
        output_y = model(batch_y)

        loss = onnr_loss(output_x, output_y, alpha)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    return loss_sum/epoch


def test_onnr_batch(batch_x, batch_y, model_1, model_2):

    assert batch_x.shape == batch_y.shape, print("batch_x and batch_y must have the same shape")

    model_1.eval()
    model_2.eval()

    with torch.no_grad():
        
        output_y_m1 = model_1(batch_y)
        output_x_m2 = model_2(batch_x)
        score = output_y_m1.mean() + output_x_m2.mean() - 2

    return score


def onnr_statistic(hidden_dims, pilot_X, arrival_Y, alpha, stride, window_size, learning_rate, epoch, device="cpu"):

    maxT, dim = arrival_Y.shape

    # init model
    model1 = MLP(dim, hidden_dims, 1).to(device)
    model2 = MLP(dim, hidden_dims, 1).to(device)

    # optimization parameter
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=learning_rate)    
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=learning_rate)    

    # placeholder for record
    wt = np.zeros(maxT)
    loss_record = np.zeros(maxT)

    # shuffle pilot_X
    x_random_idx = random.sample(range(pilot_X.shape[0]), pilot_X.shape[0]) 
    pilot_X = pilot_X[x_random_idx]

    num_batch = math.ceil(maxT/stride)

    for i in tqdm(range(num_batch)):

        end = min((i + 1) * stride, maxT)

        if end < window_size:
            x_batch = pilot_X[:end]
            y_batch = arrival_Y[:end]
        else:
            x_batch = pilot_X[np.mod(range(end - window_size, end), pilot_X.shape[0]), :]
            y_batch = arrival_Y[end - window_size:end]

        x_batch = torch.tensor(x_batch).to(device)
        y_batch = torch.tensor(y_batch).to(device)
        onnr_stat = test_onnr_batch(x_batch, y_batch, model1, model2)
        loss_1 = train_onnr_batch(x_batch, y_batch, model1, optimizer1, epoch, alpha)
        loss_2 = train_onnr_batch(y_batch, x_batch, model2, optimizer2, epoch, alpha)

        loss_record[end-1] = loss_1+loss_2
        wt[end-1] = onnr_stat
        
    return wt, loss_record


def run_onnr(hidden_dims: list, window_size: int, alpha: float, stride: int, learning_rate: float, epoch: int,
             f0_length: int, f1_length: int, burnin_length: int, 
             f0_sequence, f1_sequence, iter_num: int, save_dir: str, device: str):
    
    # reference sequence: (f0_length + f1_length) construced from f0_sequence
    # online sequence: (f0_length) construced from f0_sequence, (f1_length) construced from f1_sequence
    entire_sequence_len = burnin_length + f0_length + f1_length
    f0_with_burnin_length = f0_length + burnin_length
    f0_chunk_size = 2*f0_with_burnin_length+f1_length 
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

        y = np.float32(np.concatenate([x_chunk[:f0_with_burnin_length], y_chunk]))
        x = np.float32(x_chunk[f0_with_burnin_length:])

        Wt_h, loss_record = onnr_statistic(hidden_dims, x, y, alpha, stride, window_size, learning_rate, epoch, device)
        stat_record[i,:] = Wt_h

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 
                         "onnr_iter{}_pre{}_post{}_b{}_d{}_l{}_a{}_s{}_w{}_e{}".format(iter_num, f0_length, f1_length, burnin_length, 
                                                                                   hidden_dims[0], len(hidden_dims), alpha,
                                                                                   stride, window_size, epoch)),
                                                                                   stat_record)
    