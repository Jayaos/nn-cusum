import numpy as np
import pandas as pd
import os

import ruptures
import ruptures.metrics
from joblib import Parallel, delayed


def get_d_score(x):
    d_score = []
    for i in range(2, len(x)):
        i_d = np.abs((x[i]-x[i-1]) / (x[i-1] - x[i-2] + 10**-6))
        if i_d == 0:
            i_d = 2
        d_score.append(i_d)
    d_score = [d_score[0]] + d_score + [d_score[-1]]
    return np.array(d_score)




class RupturesBinseg(object):
    
    def __init__(self, model='rbf', jump=10):
        self.model = model
        self.jump = jump
    
    def run_predictions(self, X, n_bkps):
        algo = ruptures.Binseg(model=self.model, min_size=2, jump=self.jump).fit(X)
        my_bkps = algo.predict(n_bkps=n_bkps)
        return my_bkps
    
    def compute_gain(self, bkps, my_bkps):
        ri = ruptures.metrics.randindex(bkps, my_bkps)
        return ri
        
    
    def predict(self, X, bkps=None):
        self.n_bkps = np.arange(1, 41, 1)
        self.gains = []
        self.reco_bkps = []
        self.d_score = []
        
        # run BinSeg models with different n_bkps
        self.reco_bkps = Parallel(n_jobs=-1)(delayed(self.run_predictions)(X, i) for i in self.n_bkps)
        self.reco_bkps = np.array(self.reco_bkps)
        
        # compute gain for the found bkps
        cost = ruptures.costs.CostRbf().fit(X)
        self.gains = Parallel(n_jobs=-1)(delayed(self.compute_gain)(bkps, i) for i in self.reco_bkps)
        self.gains = np.array(self.gains)
        
        
        # select the best solution
        best_bkps = self.reco_bkps[self.gains == self.gains.max()][0]
        
        # gen score
        score = np.zeros(len(X))
        score[best_bkps[:-1]] = 1
        
        return score, best_bkps
    
    

class RupturesPelt(object):
    
    def __init__(self, model='rbf', jump=10):
        self.model = model
        self.jump = jump
    
    def run_predictions(self, X, pen):
        algo = ruptures.Pelt(model=self.model, min_size=2, jump=self.jump).fit(X)
        my_bkps = algo.predict(pen=pen)
        return my_bkps
    
    def compute_gain(self, bkps, my_bkps):
        ri = ruptures.metrics.randindex(bkps, my_bkps)
        return ri
        
    
    def predict(self, X, bkps=None):
        self.penalties = np.linspace(0, 10, 20)
        self.gains = []
        self.reco_bkps = []
        self.n_bkps = []
        
        # run Pelt models with different n_bkps
        self.reco_bkps = Parallel(n_jobs=-1)(delayed(self.run_predictions)(X, i) for i in self.penalties)
        self.reco_bkps = np.array(self.reco_bkps)
        self.n_bkps = np.array([len(i) for i in self.reco_bkps])
        #print("self.n_bkps: ", self.n_bkps)
        
        # compute gain for the found bkps
        cost = ruptures.costs.CostRbf().fit(X)
        self.gains = Parallel(n_jobs=-1)(delayed(self.compute_gain)(bkps, i) for i in self.reco_bkps)
        self.gains = np.array(self.gains)
        
        
        # select the best solution
        best_bkps = self.reco_bkps[self.gains == self.gains.max()][0]
                
        # gen score
        score = np.zeros(len(X))
        score[best_bkps[:-1]] = 1

        return score, best_bkps
    
    
    
class RupturesWindow(object):
    
    def __init__(self, model='rbf', jump=10, width=100):
        self.model = model
        self.jump = jump
        self.width = width
    
    def run_predictions(self, X, n_bkps):
        algo = ruptures.Window(model=self.model, width=self.width, min_size=2, jump=self.jump).fit(X)
        my_bkps = algo.predict(n_bkps=n_bkps)
        return my_bkps
    
    def compute_gain(self, bkps, my_bkps):
        ri = ruptures.metrics.randindex(bkps, my_bkps)
        return ri
        
    
    def predict(self, X, bkps=None):
        self.n_bkps = np.arange(1, 41, 1)
        self.gains = []
        self.reco_bkps = []
        self.d_score = []
        
        # run BinSeg models with different n_bkps
        self.reco_bkps = Parallel(n_jobs=-1)(delayed(self.run_predictions)(X, i) for i in self.n_bkps)
        self.reco_bkps = np.array(self.reco_bkps)
        
        # compute gain for the found bkps
        cost = ruptures.costs.CostRbf().fit(X)
        self.gains = Parallel(n_jobs=-1)(delayed(self.compute_gain)(bkps, i) for i in self.reco_bkps)
        self.gains = np.array(self.gains)
        
        
        # select the best solution
        best_bkps = self.reco_bkps[self.gains == self.gains.max()][0]
        
        # gen score
        score = np.zeros(len(X))
        score[best_bkps[:-1]] = 1
        
        return score, best_bkps
    
    
    
    
from online_change_clf import ChangePointDetectionOnline 
import itertools


class OnlineCLF(object):
    
    def __init__(self, lag_size=100, height=None, smooth=True):
        
        self.lag_size = lag_size
        self.height = height
        self.smooth = smooth
        
        self.params = itertools.product([1, 10], [1, 10], [1, 10], [0.01, 0.1], [1])
    
    def run_predictions(self, X, params):
        algo = ChangePointDetectionOnline(net='auto', scaler='auto', metric="KL_sym", lag_size=self.lag_size, 
                                         batch_size=params[0], step=params[1], n_epochs=params[2], lr=params[3], 
                                          periods=params[4] , lam=0.0, optimizer='Adam')
        # Detect change points
        score, my_bkps = algo.predict(X, height=self.height, smooth=self.smooth)
        if len(my_bkps) == 0:
            my_bkps = list(my_bkps) + [len(X)]
        if my_bkps[-1] != len(X):
            my_bkps = list(my_bkps) + [len(X)]
        return score, my_bkps
    
    def compute_gain(self, bkps, my_bkps):
        ri = ruptures.metrics.randindex(bkps, my_bkps)
        return ri
        
    
    def predict(self, X, bkps=None):
        self.gains = []
        self.reco_bkps = []
        self.d_score = []
        
        # run BinSeg models with different n_bkps
        self.outputs = Parallel(n_jobs=-1)(delayed(self.run_predictions)(X, i) for i in self.params)
        self.scores = np.array([i[0] for i in self.outputs])
        self.reco_bkps = np.array([i[1] for i in self.outputs])
        self.n_bkps = np.array([len(i) for i in self.reco_bkps])
        
        
        # compute gain for the found bkps
        self.gains = Parallel(n_jobs=-1)(delayed(self.compute_gain)(bkps, i) for i in self.reco_bkps)
        self.gains = np.array(self.gains)
        
        
        # select the best solution
        best_bkps = self.reco_bkps[self.gains == self.gains.max()][0]
        best_score = self.scores[self.gains == self.gains.max()][0]
        
        return best_score, best_bkps
    
    
    
from online_change_rulsif import ChangePointDetectionOnline_RuLSIF 

class OnlineRuLSIF(object):
    
    def __init__(self, lag_size=100, height=None, smooth=True):
        
        self.lag_size = lag_size
        self.height = height
        self.smooth = smooth
        
        self.params = itertools.product([1, 10], [1, 10], [1, 10], [0.01, 0.1], [1])
    
    def run_predictions(self, X, params):
        algo = ChangePointDetectionOnline_RuLSIF(net='auto', scaler='auto', alpha=0.1, metric="None", lag_size=self.lag_size, 
                                         batch_size=params[0], step=params[1], n_epochs=params[2], lr=params[3], 
                                          periods=params[4] , lam=0.0, optimizer='Adam')
        # Detect change points
        score, my_bkps = algo.predict(X, height=self.height, smooth=self.smooth)
        if len(my_bkps) == 0:
            my_bkps = list(my_bkps) + [len(X)]
        if my_bkps[-1] != len(X):
            my_bkps = list(my_bkps) + [len(X)]
        return score, my_bkps
    
    def compute_gain(self, bkps, my_bkps):
        ri = ruptures.metrics.randindex(bkps, my_bkps)
        return ri
        
    
    def predict(self, X, bkps=None):
        self.gains = []
        self.reco_bkps = []
        self.d_score = []
        
        # run BinSeg models with different n_bkps
        self.outputs = Parallel(n_jobs=-1)(delayed(self.run_predictions)(X, i) for i in self.params)
        self.scores = np.array([i[0] for i in self.outputs])
        self.reco_bkps = np.array([i[1] for i in self.outputs])
        self.n_bkps = np.array([len(i) for i in self.reco_bkps])
        
        
        # compute gain for the found bkps
        self.gains = Parallel(n_jobs=-1)(delayed(self.compute_gain)(bkps, i) for i in self.reco_bkps)
        self.gains = np.array(self.gains)
        
        
        # select the best solution
        best_bkps = self.reco_bkps[self.gains == self.gains.max()][0]
        best_score = self.scores[self.gains == self.gains.max()][0]
        
        return best_score, best_bkps
    
    
from algorithms import ChangePointDetectionRuLSIF

class OrigRuLSIF(object):
    
    def __init__(self, alpha=0.1, kernel_num=10, periods=1, window_size=100, 
                 step=10, n_runs=1, debug=0, height=None, smooth=True):
        
        self.cpd = ChangePointDetectionRuLSIF(alpha=alpha, kernel_num=kernel_num, periods=periods, 
                                              window_size=window_size, step=step, n_runs=n_runs, debug=debug)
        self.height = height
        self.smooth = smooth
        
    def predict(self, X, bkps=None):
        
        # Detect change points
        score, my_bkps = self.cpd.predict(X, height=self.height, smooth=self.smooth)
        if len(my_bkps) == 0:
            my_bkps = list(my_bkps) + [len(X)]
        if my_bkps[-1] != len(X):
            my_bkps = list(my_bkps) + [len(X)]
        return score, my_bkps
