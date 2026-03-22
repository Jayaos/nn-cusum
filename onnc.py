import numpy as np
import random
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from model import MLP
from utils.data import augment_sequence_with_replacement
import os


def train_onnc_batch(batch_x, batch_y, model, loss_fn, optimizer, epoch, device):

    assert batch_x.shape == batch_y.shape, print("batch_x and batch_y must have the same shape")
    batch_size, dim = batch_x.shape

    batch_data = torch.cat([batch_x,batch_y], axis=0).to(device)
    labels_batch = torch.cat((torch.zeros(batch_size),torch.ones(batch_size)), axis=0).to(device)

    model.train()
    loss_sum = 0.

    for e in range(epoch):
        optimizer.zero_grad()
        outputs = model(batch_data, torch.sigmoid()) 
        outputs = outputs.reshape(-1)
        loss = loss_fn(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    return loss_sum/epoch


def test_onnc_batch(batch_x, batch_y, model, device):

    assert batch_x.shape == batch_y.shape, print("batch_x and batch_y must have the same shape")
    batch_size, dim = batch_x.shape
    eps = 1e-06

    model.eval()
    with torch.no_grad():
        
        output_x = model(batch_x.to(device), torch.sigmoid())
        output_y = model(batch_y.to(device), torch.sigmoid())
        output_x = torch.clip(output_x, min=eps, max=1-eps) # why clip here?
        output_y = torch.clip(output_y, min=eps, max=1-eps) # why clip here?
        kld_score = ((1-output_x) / output_x).log10().mean() + (output_y / (1-output_y)).log10().mean()

    return kld_score


def onnc_statistic(hidden_dims, pilot_X, arrival_Y, stride, window_size, learning_rate, epoch, device="cpu"):

    maxT, dim = arrival_Y.shape

    # init model
    model = MLP(dim, hidden_dims, 1).to(device)

    # optimization parameter
    print("learning rate: {}".format(learning_rate))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)    

    # placeholder for record
    wt = np.zeros(maxT)
    loss_record = np.zeros(maxT)

    # shuffle pilot_X
    x_random_idx = random.sample(range(pilot_X.shape[0]), pilot_X.shape[0]) 
    pilot_X = pilot_X[x_random_idx]

    num_batch = int(maxT/stride)

    for i in tqdm(range(num_batch)):

        if (i+1)*stride < window_size:
            x_batch = pilot_X[:(i+1)*stride]
            y_batch = arrival_Y[:(i+1)*stride]
        else:
            x_batch = pilot_X[(np.mod(range((i+1)*stride-window_size,(i+1)*stride),pilot_X.shape[0])),:] # loop over X_tr
            y_batch = arrival_Y[(i+1)*stride-window_size:(i+1)*stride]

        x_batch = torch.Tensor(x_batch)
        y_batch = torch.Tensor(y_batch)
        loss = train_onnc_batch(x_batch, y_batch, model, nn.BCELoss(), optimizer, epoch)
        onnc_stat = test_onnc_batch(x_batch, y_batch, model=model)

        loss_record[(i+1)*stride-1] = loss
        wt[(i+1)*stride-1] = onnc_stat

    return wt, loss_record


def run_onnc(hidden_dims: list, window_size: int, stride: int, learning_rate: float, epoch: int,
             f0_length: int, f1_length: int, burnin_length: int, 
             f0_sequence, f1_sequence, iter_num: int, save_dir: str, device: str):
             

    # reference sequence: (f0_length + f1_length) construced from f0_sequence
    # online sequence: (f0_length) construced from f0_sequence, (f1_length) construced from f1_sequence
    entire_sequence_len = f0_length + f1_length
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
        
        Wt_h, loss_record = onnc_statistic(hidden_dims, x, y, stride, 
                                           window_size, learning_rate, epoch,
                                           learning_rate, [burnin_length, f0_with_burnin_length],
                                           device)
        stat_record[i,:] = Wt_h[burnin_length:]

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 
                         "onnc_iter{}_pre{}_post{}_b{}_d{}_l{}_s{}_w{}_e{}".format(iter_num, f0_length, f1_length, burnin_length, 
                                                                          hidden_dims[0], len(hidden_dims),
                                                                          stride, window_size, epoch)),
                                                                          stat_record)
    