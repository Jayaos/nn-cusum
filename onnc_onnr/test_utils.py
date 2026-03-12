import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def mean_std_report(mean, std, cols, n_digits=3):
    report = pd.DataFrame()
    for acol in cols:
        new_col_val = []
        mean_col = mean[acol].values
        std_col = std[acol].values
        for i in range(len(mean_col)):
            i_mean = np.round(mean_col[i], n_digits)
            i_std = np.round(std_col[i], n_digits)
            mean_std = str(i_mean) + ' pm ' + str(i_std)
            new_col_val.append(mean_std)
        report[acol] = new_col_val
    return report


from sklearn.preprocessing import StandardScaler
import ruptures.metrics
from copy import copy, deepcopy
import inspect


def get_label(L):
    L_new = [0]
    for i in range(1, len(L)):
        if L[i] == L[i-1]:
            L_new.append(0)
        else:
            L_new.append(1)
    return np.array(L_new)


def get_files_list(path):
    files = []
    for aname in os.listdir(path):
        if (aname[-3:] == "csv") or (aname[-3:] == "txt"): 
            files.append(os.path.join(path, aname))
    return files


def make_downsample(X, L, jump):
    states = np.cumsum(L)
    states_down = states[::jump]
    X_down      = X[::jump]
    L_down      = get_label(states_down)
    return X_down, L_down


from sklearn.preprocessing import StandardScaler
import ruptures.metrics
from test_utils import *

class Tester(object):
    
    def __init__(self):
        pass

    def run(self, afile, model, downsample=1, margin=50, verbose=0, sigma=0):

        if verbose > 0: print("File: ", afile)

        # read data
        data = pd.read_csv(afile, index_col=False)

        # decompose data
        label = data["Label"].values
        X = data.drop(columns=["Time", "Label"]).values
        X = np.nan_to_num(X)
        if downsample > 1:
            X, label = make_downsample(X, label, downsample)
        T = np.arange(len(X))

        # apply standard scaler
        X = StandardScaler().fit_transform(X)
        if verbose > 0: print("X shape: ", X.shape)

        # get true CP ids
        y_true = list(T[label == 1]) + [len(X)]
        if verbose > 0: print("y_true: ", y_true)
            
        # add noise
        if sigma > 0:
            X += np.random.normal(0, sigma, X.shape) 
        X = StandardScaler().fit_transform(X)

        # run CPD model
        score, y_pred = model.predict(X, y_true)
        if verbose > 0: print("y_pred: ", y_pred)

        # quality metrics
        ri = ruptures.metrics.randindex(y_true, y_pred)
        precision, recall = ruptures.metrics.precision_recall(y_true, y_pred, margin=margin)
        f1 = 2 * precision * recall / (precision + recall + 10**-6)
        
        self.T = T
        self.X = X
        self.label = label
        self.bkps = y_true
        
        self.score = score
        self.my_bkps = y_pred
        self.RI = ri
        self.F1 = f1
        self.precision = precision
        self.recall = recall
        self.model = model

        return


def run_test_on_dataset(dir_path, model, downsample=1, margin=50, verbose=0, sigma=0):
    
    if verbose > 0: print("Dataset: ", dir_path)
    
    # get list of files in dataset
    files = get_files_list(dir_path)
    files.sort()
    
    # fataframe for quality metrics
    report = pd.DataFrame(columns=["RI", "F1"])
    
    # run test on each file
    for afile in files:
        t = Tester()
        t.run(afile, copy(model), downsample, margin, verbose, sigma)
        report.loc[len(report)] = [t.RI, t.F1]
    
    return pd.concat([report.mean(), report.std()], axis=1)