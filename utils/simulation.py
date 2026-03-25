import numpy as np


def gen_mean_shift_P(dim,maxT):
    mu_mx = np.zeros(dim)
    sigma_mx = np.identity(dim)
    X = np.random.multivariate_normal(mu_mx, sigma_mx, maxT)
    return np.float32(X)


def gen_mean_shift_Q(dim, maxT, delta):
    mu_mx = np.zeros(dim)
    for i in range(min(dim,3)):
            mu_mx[i] = delta/(i+1)
    sigma_mx = np.identity(dim)
    X = np.random.multivariate_normal(mu_mx, sigma_mx, maxT)
    return np.float32(X)


def gen_Gamma(shape, scale, dim, n):
    X = np.random.gamma(shape, scale, size=[n,dim])
    return np.float32(X)


def compute_arl_edd(data, maxT, nchange, nrun, ARL):
    # raw data shape: nrun * maxT
    
    pre_change_data = data[:, :nchange]
    post_change_data = data[:, nchange:]
    
    max_vals = np.max(pre_change_data, axis=1) # nrun
    ngrid = len(ARL)
    threshold_list = np.zeros(ngrid)
    
    # p = P(max>b)
    hat_p = 1-np.exp((-nchange)/ARL)
    threshold_list = np.quantile(max_vals,1-hat_p)
    
    # EDD
    EDD = np.zeros(ngrid)
    
    for i in range(ngrid):
        threshold = threshold_list[i]
        DD_per_threshold = np.zeros(nrun)
        
        for r in range(nrun):
            try:
                DD_per_threshold[r] = np.argwhere(post_change_data[r] >= threshold)[0]
            except:
                # in case fail to detect since we assume H1 ~ post change dist, max DD should be maxT-nchange
                DD_per_threshold[r] = DD = maxT-nchange
        
        EDD[i] = np.mean(DD_per_threshold) # no std calculation 
        
    return ARL, EDD
    # return ARL, DD_per_threshold
