import numpy as np


def calculate_arl_edd(stat_record, arl_list, cp_location):
    
    arl_list = np.array(arl_list)
    iter_num = stat_record.shape[0]
    online_size = stat_record.shape[1]-cp_location
    
    hat_p_list = 1-np.exp((-cp_location)/arl_list)
    h0_stat_record = stat_record[:,:cp_location]
    h1_stat_record = stat_record[:,cp_location:]
    h0_stat_record_max = np.max(h0_stat_record, axis=1)
    thresholds = np.quantile(h0_stat_record_max, 1-hat_p_list)
    arl_edd_record = np.zeros((iter_num, len(arl_list)))

    for i, threshold in enumerate(thresholds):
        for j in range(iter_num):
            try:
                arl_edd_record[j,i] = np.argwhere(h1_stat_record[j,:] >= threshold)[0][0]
            except:
                arl_edd_record[j,i] = online_size

    return arl_edd_record


def calculate_type1err_edd(stat_record, alpha_list, cp_location):

    alpha_list = np.array(alpha_list)
    iter_num = stat_record.shape[0]
    online_size = stat_record.shape[1]-cp_location
    h0_stat_record = stat_record[:,:cp_location]
    h1_stat_record = stat_record[:,cp_location:]
    h0_stat_record_max = np.max(h0_stat_record, axis=1)
    thresholds = np.percentile(h0_stat_record_max, 100-alpha_list)
    
    type1err_edd_record = np.zeros((iter_num, len(alpha_list)))

    for i, threshold in enumerate(thresholds):
        for j in range(iter_num):
            try:
                type1err_edd_record[j,i] = np.argwhere(h1_stat_record[j,:] >= threshold)[0][0]
            except:
                type1err_edd_record[j,i] = online_size
                
    return type1err_edd_record


def count_type1err_success(stat_record, alpha_list, cp_location):

    iter_num, seq_len = stat_record.shape
    f1_len = seq_len - cp_location
    
    type1err_edd_record = calculate_type1err_edd(stat_record, alpha_list, cp_location)
    tf_arr = type1err_edd_record != f1_len
    colsum = np.sum(tf_arr, axis=0)
    
    for ind, sums in zip(alpha_list, colsum):
        print("{} : {} successes".format(ind/100, sums))