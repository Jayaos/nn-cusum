import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


# utils
def autoregression_matrix(X, periods=1, fill_value=0):
    shifted_x = [pd.DataFrame(X).shift(periods=i, fill_value=fill_value).values for i in range(periods)]
    X_auto = np.hstack(tuple(shifted_x))
    return X_auto


def reference_test(X, window_size=2, step=1):
    T = []
    reference = []
    test = []
    for i in range(2*window_size-1, len(X), step):
        T.append(i)
        reference.append(X[i-2*window_size+1:i-window_size+1])
        test.append(X[i-window_size+1:i+1])
    return np.array(T), np.array(reference), np.array(test)



def KL_score_unsym(ref_ratios, test_ratios):
    score = np.mean(np.log(test_ratios))
    return score

def KL_score(ref_ratios, test_ratios):
    score = KL_score_unsym(ref_ratios, test_ratios) + KL_score_unsym(1./test_ratios, 1./ref_ratios)
    return score


def PE_score_unsym(ref_ratios, test_ratios, alpha=0.):
    score = (-0.5 *       alpha  * np.mean(test_ratios**2)) + \
            (-0.5 * (1. - alpha) * np.mean(ref_ratios**2))  + np.mean(test_ratios) - 0.5
    return score

def PE_score(ref_ratios, test_ratios, alpha=0.):
    score = PE_score_unsym(ref_ratios, test_ratios, alpha)# - PE_score_unsym(test_ratios, ref_ratios, alpha)
    return score


def KL(ref_preds, test_preds):
    return np.mean(np.log(test_preds + 10**-3)) - np.mean(np.log(1. - test_preds + 10**-3))

def KL_sym(ref_preds, test_preds):
    return np.mean(np.log(test_preds + 10**-3))     - np.mean(np.log(1. - test_preds + 10**-3)) + \
           np.mean(np.log(1. - ref_preds + 10**-3)) - np.mean(np.log(ref_preds + 10**-3))

def JSD(ref_preds, test_preds):
    return np.log(2) + 0.5 * np.mean(np.log(test_preds + 10**-3)) + 0.5 * np.mean(np.log(1. - ref_preds + 10**-3))

def PE(ref_preds, test_preds):
    scores = test_preds / (1. - test_preds + 10**-6) - 1.
    scores = np.clip(scores, 0, 1000)
    return np.mean(scores)

def PE_sym(ref_preds, test_preds):
    scores_1 = test_preds / (1. - test_preds + 10**-6) - 1.
    scores_1 = np.clip(scores_1, 0, 1000)
    scores_2 = (1. - ref_preds) / (ref_preds + 10**-6) - 1.
    scores_2 = np.clip(scores_2, 0, 1000)
    return np.mean(scores_1) + np.mean(scores_2)

def Wasserstein(ref_preds, test_preds):
    return np.mean(test_preds) - np.mean(ref_preds)



from sklearn.model_selection import train_test_split
from copy import deepcopy
from joblib import Parallel, delayed


from scipy import interpolate
from scipy.signal import find_peaks
def unified_score(T, T_score, score):
    uni_score = np.zeros(len(T))
    inter = interpolate.interp1d(T_score, score, kind='previous', fill_value=(0, 0), bounds_error=False)
    uni_score = inter(T)
    return uni_score


# RuLSIF
class ChangePointDetectionRuLSIF(object):
    
    def __init__(self, alpha=0.1, kernel_num=100, 
                 periods=1, window_size=100, step=1, n_runs=1, debug=0):
        self.alpha = alpha
        self.kernel_num = kernel_num
        self.periods = periods
        self.window_size = window_size
        self.step = step
        self.n_runs = n_runs
        self.debug = debug
        
        
    def densration_gridsearch(self, X_ref, X_test):
        lambda_range = 10**np.linspace(-3, 3, 7) #np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1])
        sigma_range  = 10**np.linspace(-3, 3, 25) #np.array([10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3])        
        estimator_1 = densratio(X_ref, X_test, self.alpha, sigma_range, lambda_range, self.kernel_num, verbose=False)
        estimator_2 = densratio(X_test, X_ref, self.alpha, sigma_range, lambda_range, self.kernel_num, verbose=False)
        w1_ref = estimator_1.compute_density_ratio(X_ref)
        w2_test = estimator_2.compute_density_ratio(X_test)
        score_max = (0.5 * np.mean(w1_ref) - 0.5) + (0.5 * np.mean(w2_test) - 0.5)
        return score_max
        
        
    def reference_test_predict(self, X_ref, X_test):
        score = self.densration_gridsearch(X_ref, X_test)
        return score
    
    
    def reference_test_predict_n_times(self, X_ref, X_test):
        scores = []
        for i in range(self.n_runs):
            ascore = self.reference_test_predict(X_ref, X_test)
            scores.append(ascore)
        return np.mean(scores)
        
    
    def predict(self, X, distance=5, height=None, smooth=False):
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = reference_test(X_auto, window_size=self.window_size, step=1)
        scores = []
        T_scores = []
        iters = range(0, len(reference), self.step)
        scores = Parallel(n_jobs=-1)(delayed(self.reference_test_predict_n_times)(reference[i], test[i]) for i in iters)
        T_scores = np.array([T[i] for i in iters])
        
        T = np.arange(len(X))
        scores = unified_score(T, T_scores-self.step, scores)
        
        shift = self.window_size
        scores = unified_score(T, T-shift, scores)
        
        if smooth:
            from scipy.signal import savgol_filter
            width = int((np.round(0.25 * self.window_size) // 2) * 2 + 1)
            scores = savgol_filter(scores, width, 1)
        
        width = 0.25 * (self.window_size)
        peaks, _ = find_peaks(scores, distance=distance, width=width, height=height)
        
        return np.array(scores), peaks
    
    

# # KLIEP
# from pykliep import DensityRatioEstimator 


# class ChangePointDetectionKLIEP(object):
    
#     def __init__(self, num_params, sigmas, metric="KL", periods=1, window_size=100, step=1, n_runs=10, debug=0):
#         self.num_params = num_params
#         self.sigmas = sigmas
#         self.metric = metric
#         self.periods = periods
#         self.window_size = window_size
#         self.step = step
#         self.n_runs = n_runs
#         self.debug = debug
        
        
#     def kliep_gridsearch(self, X_ref, X_test):
#         score_max = -999.
#         for sigma in self.sigmas:
#             for num in self.num_params:
#                 estimator_1 = DensityRatioEstimator(max_iter=1000, num_params=[num], cv=2, sigmas=[sigma])
#                 estimator_1.fit(X_ref, X_test)
#                 estimator_2 = DensityRatioEstimator(max_iter=1000, num_params=[num], cv=2, sigmas=[sigma])
#                 estimator_2.fit(X_test, X_ref)
#                 if self.metric == "KL":
#                     score = estimator_1.score(X_test) + estimator_2.score(X_ref)
#                 else:
#                     score = 0
#                 if score >= score_max:
#                     score_max = score
#         return score_max
        
        
#     def reference_test_predict(self, X_ref, X_test):
#         score = self.kliep_gridsearch(X_ref, X_test)
#         return score
    
    
#     def reference_test_predict_n_times(self, X_ref, X_test):
#         scores = []
#         for i in range(self.n_runs):
#             ascore = self.reference_test_predict(X_ref, X_test)
#             scores.append(ascore)
#         return np.mean(scores)
        
    
#     def predict(self, X):
#         X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
#         T, reference, test = reference_test(X_auto, window_size=self.window_size, step=1)
#         scores = []
#         T_scores = []
#         iters = range(0, len(reference), self.step)
#         scores = Parallel(n_jobs=-1)(delayed(self.reference_test_predict_n_times)(reference[i], test[i]) for i in iters)
#         T_scores = [T[i] for i in iters]
#         return np.array(T_scores), np.array(scores)
    
    
    
# Classification
class ChangePointDetectionClassifier(object):
    
    def __init__(self, base_classifier, metric="KL", periods=1, window_size=100, step=1, n_runs=10, debug=0):
        self.base_classifier = base_classifier
        self.metric = metric
        self.periods = periods
        self.window_size = window_size
        self.step = step
        self.n_runs = n_runs
        self.debug = debug
        
    def densratio(self, y_pred):
        w = (y_pred + 10**-3) / (1 - y_pred + 10**-3)
        return w
        
        
    def reference_test_predict(self, X_ref, X_test):
        y_ref = np.zeros(len(X_ref))
        y_test = np.ones(len(X_test))
        X = np.vstack((X_ref, X_test))
        y = np.hstack((y_ref, y_test))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, 
                                                            stratify=y, random_state=np.random.randint(0, 1000))
        classifier = deepcopy(self.base_classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        ref_preds = y_pred[y_test == 0]
        test_preds = y_pred[y_test == 1]
        ratios = self.densratio(y_pred)
        ref_ratios = ratios[y_test == 0]
        test_ratios = ratios[y_test == 1]
        if self.metric == "KL":
            score = KL(ref_preds, test_preds)
        elif self.metric == "KL_sym":
            score = KL_sym(ref_preds, test_preds)
        elif self.metric == "JSD":
            score = JSD(ref_preds, test_preds)
        elif self.metric == "PE":
            score = PE(ref_preds, test_preds)
        elif self.metric == "PE_sym":
            score = PE_sym(ref_preds, test_preds)
        elif self.metric == "W":
            score = Wasserstein(ref_preds, test_preds)
        elif self.metric == "ROCAUC":
            score = roc_auc_score(y_test, y_pred) - 0.5
        else:
            score = 0
        return score
    
    
    def reference_test_predict_n_times(self, X_ref, X_test):
        scores = []
        for i in range(self.n_runs):
            ascore = self.reference_test_predict(X_ref, X_test)
            scores.append(ascore)
        return np.mean(scores)
        
    
    def predict(self, X):
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = reference_test(X_auto, window_size=self.window_size, step=1)
        scores = []
        T_scores = []
        iters = range(0, len(reference), self.step)
        scores = Parallel(n_jobs=-1)(delayed(self.reference_test_predict_n_times)(reference[i], test[i]) for i in iters)
        T_scores = [T[i] for i in iters]
        return np.array(T_scores), np.array(scores)
    
    
    
# NN RuLSIF
class ChangePointDetectionClassifier_RuLSIF(object):
    
    def __init__(self, base_classifier, metric="KL", periods=1, window_size=100, step=1, n_runs=10, debug=0):
        self.base_classifier = base_classifier
        self.metric = metric
        self.periods = periods
        self.window_size = window_size
        self.step = step
        self.n_runs = n_runs
        self.debug = debug
        
        
    def reference_test_predict(self, X_ref, X_test):
        y_ref = np.zeros(len(X_ref))
        y_test = np.ones(len(X_test))
        X = np.vstack((X_ref, X_test))
        y = np.hstack((y_ref, y_test))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, 
                                                            stratify=y, random_state=np.random.randint(0, 1000))
        
        classifier_1 = deepcopy(self.base_classifier)
        classifier_1.fit(X_train, y_train)
        ratios = classifier_1.predict_proba(X_test)
        ref_ratios = ratios[y_test == 0]
        test_ratios = ratios[y_test == 1]
        if self.metric == "PE":
            score_1 = 0.5 * np.mean(test_ratios) - 0.5 # PE_score_unsym(ref_ratios, test_ratios, classifier_1.alpha) - PE_score_unsym(test_ratios, ref_ratios, classifier_1.alpha)
        else:
            score_1 = 0
        
        classifier_2 = deepcopy(self.base_classifier)
        classifier_2.fit(X_train, (1-y_train))
        ratios = classifier_2.predict_proba(X_test)
        ref_ratios = ratios[(1-y_test) == 0]
        test_ratios = ratios[(1-y_test) == 1]
        if self.metric == "PE":
            score_2 = 0.5 * np.mean(test_ratios) - 0.5 # PE_score_unsym(ref_ratios, test_ratios, classifier_2.alpha) - PE_score_unsym(test_ratios, ref_ratios, classifier_2.alpha)
        else:
            score_2 = 0
        score = score_1 + score_2
        
        return score
    
    
    def reference_test_predict_n_times(self, X_ref, X_test):
        scores = []
        for i in range(self.n_runs):
            ascore = self.reference_test_predict(X_ref, X_test)
            scores.append(ascore)
        return np.mean(scores)
        
    
    def predict(self, X):
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = reference_test(X_auto, window_size=self.window_size, step=1)
        scores = []
        T_scores = []
        iters = range(0, len(reference), self.step)
        scores = Parallel(n_jobs=-1)(delayed(self.reference_test_predict_n_times)(reference[i], test[i]) for i in iters)
        T_scores = [T[i] for i in iters]
        scores = np.array(scores)
        #scores = np.nan_to_num(scores)
        return np.array(T_scores), scores



# NN Exp
class ChangePointDetectionClassifier_Exp(object):
    
    def __init__(self, base_classifier, metric="Exp", periods=1, window_size=100, step=1, n_runs=10, debug=0):
        self.base_classifier = base_classifier
        self.metric = metric
        self.periods = periods
        self.window_size = window_size
        self.step = step
        self.n_runs = n_runs
        self.debug = debug
        
    def densratio(self, y_pred):
        w = (y_pred + 10**-3) / (1 - y_pred + 10**-3)
        return w
        
        
    def reference_test_predict(self, X_ref, X_test):
        y_ref = np.zeros(len(X_ref))
        y_test = np.ones(len(X_test))
        X = np.vstack((X_ref, X_test))
        y = np.hstack((y_ref, y_test))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, 
                                                            stratify=y, random_state=np.random.randint(0, 1000))
        classifier = deepcopy(self.base_classifier)
        classifier.fit(X_train, y_train)
        log_ratios = classifier.predict_proba(X_test)
        ref_log_ratios = log_ratios[y_test == 0]
        test_log_ratios = log_ratios[y_test == 1]
        if self.metric == "KL":
            score = test_log_ratios.mean()
        elif self.metric == "PE":
            score = PE_score(np.exp(ref_log_ratios), np.exp(test_log_ratios), classifier.alpha)
        elif self.metric == "Exp":
            score = np.exp(- (2 * y_test - 1) * log_ratios).mean() - 1
        else:
            score = 0
        return score
    
    
    def reference_test_predict_n_times(self, X_ref, X_test):
        scores = []
        for i in range(self.n_runs):
            ascore = self.reference_test_predict(X_ref, X_test)
            scores.append(ascore)
        return np.mean(scores)
        
    
    def predict(self, X):
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        T, reference, test = reference_test(X_auto, window_size=self.window_size, step=1)
        scores = []
        T_scores = []
        iters = range(0, len(reference), self.step)
        scores = Parallel(n_jobs=-1)(delayed(self.reference_test_predict_n_times)(reference[i], test[i]) for i in iters)
        T_scores = [T[i] for i in iters]
        scores = np.array(scores)
        return np.array(T_scores), scores

    

# RNN

def autoregression_matrix_rnn(X, periods=1, fill_value=0):
    shifted_x = [pd.DataFrame(X).shift(periods=i, fill_value=fill_value).values for i in range(periods)]
    X_auto = np.array(shifted_x[::-1])
    return X_auto

def reference_test_rnn(X, window_size=2, step=1):
    T = []
    reference = []
    test = []
    for i in range(2*window_size-1, X.shape[1], step):
        T.append(i)
        reference.append(X[:, i-2*window_size+1:i-window_size+1, :])
        test.append(X[:, i-window_size+1:i+1, :])
    return np.array(T), np.array(reference), np.array(test)


class ChangePointDetection_RNN(object):
    
    def __init__(self, base_classifier, metric="KL", periods=1, window_size=100, step=1, n_runs=10, debug=0):
        self.base_classifier = base_classifier
        self.metric = metric
        self.periods = periods
        self.window_size = window_size
        self.step = step
        self.n_runs = n_runs
        self.debug = debug
        
    def densratio(self, y_pred):
        w = (y_pred + 10**-3) / (1 - y_pred + 10**-3)
        return w
        
        
    def reference_test_predict(self, X_ref, X_test):
        y_ref = np.zeros(X_ref.shape[1])
        y_test = np.ones(X_test.shape[1])
        X = np.hstack((X_ref, X_test))
        y = np.hstack((y_ref, y_test))
        indx = np.arange(len(y))
        indx_train, indx_test = train_test_split(indx, test_size=0.5, 
                                                            stratify=y, random_state=np.random.randint(0, 1000))
        X_train = X[:, indx_train, :]
        X_test = X[:, indx_test, :]
        y_train = y[indx_train]
        y_test = y[indx_test]
        classifier = deepcopy(self.base_classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        ref_preds = y_pred[y_test == 0]
        test_preds = y_pred[y_test == 1]
        ratios = self.densratio(y_pred)
        ref_ratios = ratios[y_test == 0]
        test_ratios = ratios[y_test == 1]
        if self.metric == "KL":
            score = KL(ref_preds, test_preds)
        elif self.metric == "KL_sym":
            score = KL_sym(ref_preds, test_preds)
        elif self.metric == "JSD":
            score = JSD(ref_preds, test_preds)
        elif self.metric == "PE":
            score = PE(ref_preds, test_preds)
        elif self.metric == "PE_sym":
            score = PE_sym(ref_preds, test_preds)
        elif self.metric == "W":
            score = Wasserstein(ref_preds, test_preds)
        elif self.metric == "ROCAUC":
            score = roc_auc_score(y_test, y_pred) - 0.5
        else:
            score = 0
        return score
    
    
    def reference_test_predict_n_times(self, X_ref, X_test):
        scores = []
        for i in range(self.n_runs):
            ascore = self.reference_test_predict(X_ref, X_test)
            scores.append(ascore)
        return np.mean(scores)
        
    
    def predict(self, X):
        X_auto = autoregression_matrix_rnn(X, periods=self.periods, fill_value=0)
        T, reference, test = reference_test_rnn(X_auto, window_size=self.window_size, step=1)
        scores = []
        T_scores = []
        iters = range(0, len(reference), self.step)
        scores = Parallel(n_jobs=-1)(delayed(self.reference_test_predict_n_times)(reference[i], test[i]) for i in iters)
        T_scores = [T[i] for i in iters]
        return np.array(T_scores), np.array(scores)