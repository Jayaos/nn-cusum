import numpy as np
import random
import copy
import torch
import os
from tqdm import tqdm
from torch.nn import functional as F
from model import MLP
from utils.data import augment_sequence_with_replacement


def test_statistic(hidden_dims, pilot_X, arrival_Y, stride, tr_window, te_window, batch_size, learning_rate, reset,
                   use_tr_only=False, use_Xte_window=True, device="cpu"):
    
    # training window parameters
    maxT, dim = arrival_Y.shape
    half_stride = int(stride/2)
    n_fold_tr = int(tr_window/stride) 
    n_fold_te = int(te_window/stride) #number of subwindows in test window
    
    # pilotX, random split training set (to arrive) and test set
    nXpool = pilot_X.shape[0]
    nX_te = int(te_window/2) # effective test window size
    nX_tr = nXpool-nX_te
    samplex_random_idx = random.sample(range(nXpool), nXpool ) 
    X_tr = pilot_X[samplex_random_idx[:nX_tr],:] # train
    X_te = pilot_X[samplex_random_idx[nX_tr:nX_tr+nX_te] ,:] # test
    
    # init model
    model = MLP(dim, hidden_dims, 1).to(device)
    model_init_params = copy.deepcopy(model.state_dict())

    # optimization parameter
    print("learning rate: {}".format(learning_rate))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)    
    
    #storage of training statistics, average on testing window
    mXt, mYt = np.zeros(maxT), np.zeros(maxT) #window averaged test statistic
    Wprevr =0
    dWt, Wt = np.zeros(maxT), np.zeros(maxT) #CUSUM

    model.train()
    model.load_state_dict(model_init_params)

    # tr_window and tr_window_lag
    wtr_len = 0 # the record of window len 
    dataX_wtr = np.float32( np.zeros( (n_fold_tr*half_stride,dim) ))
    dataY_wtr = np.float32( np.zeros( (n_fold_tr*half_stride,dim) ))

    wtr_lag_len =0
    dataX_wtr_lag = np.float32( np.zeros( (n_fold_tr*half_stride,dim) ))
    dataY_wtr_lag = np.float32( np.zeros( (n_fold_tr*half_stride,dim) ))

    wte_len = 0
    dataX_wte = np.float32( np.zeros( (n_fold_te*half_stride,dim) ))
    dataY_wte = np.float32( np.zeros( (n_fold_te*half_stride,dim) )) 
    #not used for testing, just go together with Yte_window

    i= 0 #index of actual time
    train_loop_count = 0

    while i <= maxT-stride:
        #load the train split of minibatch of data
        miniX_tr = X_tr[(np.mod(range(i,i+half_stride),nX_tr)),:] #loope over X_tr
        miniY_tr = arrival_Y[i:i+half_stride,:] 
        # hit the model with minibatch of training samples
        train_miniwindow(miniX_tr, miniY_tr, model, optimizer, device)

        # update training window
        if wtr_len < tr_window/2:
            augment_window_update(dataX_wtr, wtr_len, miniX_tr)
            augment_window_update(dataY_wtr, wtr_len, miniY_tr)
            wtr_len=wtr_len+half_stride
        else: #training window is full
            if use_tr_only:
                Xtr_exist = dataX_wtr[:wtr_len,:]
                Ytr_exist = dataY_wtr[:wtr_len,:]
            else:
                Xtr_exist = np.concatenate((dataX_wtr[:wtr_len,:], dataX_wtr_lag[:wtr_lag_len]),axis=0)
                Ytr_exist = np.concatenate((dataY_wtr[:wtr_len,:], dataY_wtr_lag[:wtr_lag_len]),axis=0)
            if Xtr_exist.shape[0] > 0:
                train_loop_count += 1
                train_loop(Xtr_exist, Ytr_exist, model, optimizer, batch_size, device)  
            #train loop over the exising training stack, including training window and lag training window
            # shift and update training window
            shift_window_update(dataX_wtr,miniX_tr)
            shift_window_update(dataY_wtr,miniY_tr)

        #load the test split of minibatch of data
        miniX_te = X_tr[(np.mod(range(i+half_stride,i+stride),nX_tr)),:]
        miniY_te = arrival_Y[i+half_stride:i+stride,:]
    
        # update test window
        if wte_len < te_window/2:
            # put into test window
            augment_window_update(dataX_wte, wte_len, miniX_te)
            augment_window_update(dataY_wte, wte_len, miniY_te)
            wte_len=wte_len+half_stride
        else: #if a test window is full
            # augment the training lag windo
            ##### !!!!!!!!!!!
            if wtr_lag_len < tr_window/2:
                augment_window_update(dataX_wtr_lag, wtr_lag_len, dataX_wte[0:half_stride,:])
                augment_window_update(dataY_wtr_lag, wtr_lag_len, dataY_wte[0:half_stride,:])
                wtr_lag_len=wtr_lag_len+half_stride
            else:
                # shift a stride fwd in the training lag window
                shift_window_update(dataX_wtr_lag,dataX_wte[0:half_stride,:])
                shift_window_update(dataY_wtr_lag,dataY_wte[0:half_stride,:])         
            # shift a stride fwd of test sindow
            shift_window_update(dataX_wte,miniX_te)
            shift_window_update(dataY_wte,miniY_te)

        # deploy the model on the test window
        with torch.no_grad():
            dataY_te=dataY_wte[:wte_len,:]
            uY = model(torch.tensor(dataY_te).to(device))
            uYmean = uY.reshape(-1).detach().cpu().numpy().mean()
            mYt[i+stride-1] = uYmean
            if use_Xte_window:
                dataX_te=dataX_wte[:wte_len,:]
            else:
                dataX_te= X_te #though data is the same X_te, the model differs, recompute uX each step
            uX = model(torch.tensor(dataX_te).to(device)) 
            uXmean = uX.reshape(-1).detach().cpu().numpy().mean()
            mXt[i+stride-1] = uXmean

        # recursive CUSUM
        eta_stride = uYmean-uXmean
        dWt[i+stride-1] = eta_stride
        Wt[i+stride-1] =  max(0, eta_stride+Wprevr)
        if i+stride in reset:
            Wt[i+stride-1] = 0
        Wprevr = Wt[i+stride-1]
    
        #scrable Xtr when one pass is done
        if np.mod(i, nX_tr) < stride:
            X_tr = X_tr[random.sample(range(nX_tr),nX_tr ),:]
        
        #increase the batch index, and sample index i    
        i = i+stride
    
    idx = range(stride-1, maxT, stride) 
    Wt = Wt[idx]
    dWt = dWt[idx]
    mXt = mXt[idx]
    mYt = mYt[idx]
    return idx, Wt, dWt, model, mXt, mYt


def shift_window_update(window, batch_data):
    win_len=window.shape[0]
    batch_len=batch_data.shape[0]
    
    window[:win_len-batch_len,:]=window[batch_len:win_len,:]
    window[win_len-batch_len:,:]=batch_data


def augment_window_update(window, cur_len, batch_data):
    batch_len=batch_data.shape[0]
    window[cur_len:cur_len+batch_len,:] = batch_data


def train_loop(X_tr, Y_tr, model, optimizer, batch_size, device):
    nX, dim = X_tr.shape
    #nY = Y_tr.shape[0] #assume nX=nY
    num_batch = int(nX/batch_size)
    
    #shuffle the data samples
    idx_fold = torch.randperm(nX).numpy()
    dataX = X_tr[idx_fold,:]
    dataY = Y_tr[idx_fold,:]    
    
    train_loss= 0
    for ibatch in range(num_batch):
        data_batch = torch.tensor( np.concatenate( 
                (dataX[batch_size*ibatch: batch_size*(ibatch+1),:],
                 dataY[batch_size*ibatch: batch_size*(ibatch+1),:]  ), axis=0) )
        labels_batch = torch.tensor(
                np.concatenate((np.zeros(batch_size), np.ones(batch_size)), axis=0),
                dtype=torch.long,
                device=device,
        )
        # train the model
        optimizer.zero_grad()
        # forward pass
        uxb = model(data_batch.to(device))#at least 2 samples, one from X, one from Y
        outputs = torch.cat( (-uxb/2, uxb/2),1)
        pred = F.log_softmax(outputs, dim=1)
        loss = F.nll_loss(pred, labels_batch, reduction='mean')
        #backward pass
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/nX


def train_miniwindow(miniX, miniY, model, optimizer, device):
    mini_size,dim = miniX.shape
    data_batch = torch.tensor( np.concatenate( (miniX,miniY ), axis=0) )
    labels_batch = torch.tensor(
        np.concatenate((np.zeros(mini_size), np.ones(mini_size)), axis=0),
        dtype=torch.long,
        device=device,
    )
    # train the model
    optimizer.zero_grad()
    # forward pass
    uxb = model(data_batch.to(device))#at least 2 samples, one from X, one from Y
    outputs = torch.cat( (-uxb/2, uxb/2),1)
    pred = F.log_softmax(outputs, dim=1)
    loss = F.nll_loss(pred, labels_batch, reduction='sum')
    #backward pass
    loss.backward()
    optimizer.step()

    return loss.item()/mini_size


def run_nncusum(hidden_dims: list, window_size: int, stride: int, 
                batch_size: int, learning_rate: float, 
                f0_length: int, f1_length: int, burnin_length: int, 
                f0_sequence, f1_sequence,
                iter_num: int, save_dir: str, device: str):
    
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
        
        idxt_nn, Wt_nn, dWt_nn, model, mXt_nn, mYt_nn = test_statistic(hidden_dims, x, y, stride, 
                                                                       window_size, window_size, batch_size, 
                                                                       learning_rate, [burnin_length, f0_with_burnin_length],
                                                                       device=device)
            
        stat_record[i,idxt_nn] = Wt_nn
    
    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 
                         "nncusum_iter{}_pre{}_post{}_b{}_d{}_l{}_s{}_w{}".format(iter_num, f0_length, f1_length, burnin_length, 
                                                                          hidden_dims[0], len(hidden_dims),
                                                                          stride, window_size)),
                                                                          stat_record)


def run_nncusum_separate_prepost(hidden_dims: list, window_size: int, stride: int,
                             batch_size: int, learning_rate: float,
                             f0_length: int, f1_length: int, burnin_length: int,
                             f0_sequence, f1_sequence,
                             iter_num: int, save_dir: str, device: str):

    f0_with_burnin_length = f0_length + burnin_length
    f1_with_burnin_length = f1_length + burnin_length
    f0_chunk_size = 2 * f0_with_burnin_length + f1_length
    f1_chunk_size = f1_length

    wt_pre_record = np.zeros(shape=(iter_num, f0_with_burnin_length))
    wt_post_record = np.zeros(shape=(iter_num, f1_with_burnin_length))
    wt_prepost_record = np.zeros(shape=(iter_num, f0_length + f1_length))

    if f0_sequence.shape[0] < iter_num * f0_chunk_size:
        print("f0 sequence do not have enough data")
        print("current f0 sequence length: {}, required length: {}".format(f0_sequence.shape[0], iter_num * f0_chunk_size))
        f0_sequence = augment_sequence_with_replacement(f0_sequence, iter_num * f0_chunk_size)

    if f1_sequence.shape[0] < iter_num * f1_chunk_size:
        print("f1 sequence do not have enough data")
        print("current f1 sequence length: {}, required length: {}".format(f1_sequence.shape[0], iter_num * f1_chunk_size))
        f1_sequence = augment_sequence_with_replacement(f1_sequence, iter_num * f1_chunk_size)

    for i in tqdm(range(iter_num)):

        x_chunk = f0_sequence[i * f0_chunk_size:(i + 1) * f0_chunk_size, :]
        y_chunk = f1_sequence[i * f1_chunk_size:(i + 1) * f1_chunk_size, :]

        pilot_X_full = np.float32(x_chunk[f0_with_burnin_length:])
        arrival_Y_pre = np.float32(x_chunk[:f0_with_burnin_length])
        arrival_Y_post = np.float32(np.concatenate([x_chunk[:burnin_length], y_chunk]))

        pilot_X_pre = pilot_X_full[:f0_with_burnin_length]
        pilot_X_post = pilot_X_full[:f1_with_burnin_length]

        idxt_pre, Wt_pre, _, _, _, _ = test_statistic(
            hidden_dims,
            pilot_X_pre,
            arrival_Y_pre,
            stride,
            window_size,
            window_size,
            batch_size,
            learning_rate,
            [burnin_length],
            device=device,
        )
        idxt_pre = np.asarray(list(idxt_pre))
        wt_pre_record[i, idxt_pre] = Wt_pre
        pre_mask = idxt_pre >= burnin_length
        wt_prepost_record[i, idxt_pre[pre_mask] - burnin_length] = Wt_pre[pre_mask]

        idxt_post, Wt_post, _, _, _, _ = test_statistic(
            hidden_dims,
            pilot_X_post,
            arrival_Y_post,
            stride,
            window_size,
            window_size,
            batch_size,
            learning_rate,
            [burnin_length],
            device=device,
        )

        idxt_post = np.asarray(list(idxt_post))
        wt_post_record[i, idxt_post] = Wt_post
        post_mask = idxt_post >= burnin_length
        wt_prepost_record[i, f0_length + (idxt_post[post_mask] - burnin_length)] = Wt_post[post_mask]

    print("saving results...")
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(
        save_dir,
        "nncusum_separate_seq_iter{}_pre{}_post{}_b{}_d{}_l{}_s{}_w{}".format(
            iter_num,
            f0_length,
            f1_length,
            burnin_length,
            hidden_dims[0],
            len(hidden_dims),
            stride,
            window_size,
        ),
    ), Wt_pre=wt_pre_record, Wt_post=wt_post_record, Wt_prepost=wt_prepost_record)

