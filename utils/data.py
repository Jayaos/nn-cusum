import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def higgs_data_processing():
    ...
    

def miniboone_data_processing(data_dir, saving_dir):
    
    with open(data_dir) as f:
        lines = f.readlines()
    
    signal_num = np.array(lines[0].strip("\n").split(" "))
    positive_signal_num = int(signal_num[1])
    negative_signal_num = int(signal_num[2])
    
    miniboone_data = pd.read_csv(data_dir, names=[str(x) for x in range(50)], delim_whitespace=True, skiprows=1)
    miniboone_data = miniboone_data.to_numpy()
    label_col = np.concatenate([np.ones(positive_signal_num), np.zeros(negative_signal_num)])
    label_col = np.reshape(label_col, newshape=(label_col.shape[0],1))
    print(miniboone_data.shape)
    print(label_col.shape)
    miniboone_data = np.concatenate([miniboone_data, label_col], axis=1) 
    print(miniboone_data.shape)
    
    outlier_idx = (miniboone_data[:, 0] < -100) # remove outliers having values at exactly −1000 for every column and were removed
    miniboone_data = miniboone_data[~outlier_idx]
    print(miniboone_data.shape)
    
    # Remove any features that have too many re-occuring (more than 10000) real values.
    features_to_remove = []
    for i in range(miniboone_data.shape[1]):
        
        if i == (miniboone_data.shape[1]-1):
            break
        feature = miniboone_data[:,i]
        print('Feature index: ', i)
        print('Count of the most frequent real value: ' , np.max(np.unique(feature, return_counts=True)[1]))
        max_count = np.max(np.unique(feature, return_counts=True)[1])
        if max_count > 10000:
            features_to_remove.append(i)
            print('DROP')
        else:
            print('KEEP')
        print('\n\n')
    
    miniboone_data = miniboone_data[:, np.array([i for i in range(miniboone_data.shape[1]) if i not in features_to_remove])]
    
    standardizer = StandardScaler()
    miniboone_data_standardized = standardizer.fit_transform(miniboone_data[:,:-1])
    
    signal = miniboone_data_standardized[miniboone_data[:,-1] == 1]
    background = miniboone_data_standardized[miniboone_data[:,-1] == 0]

    print("signal data : {}".format(signal.shape))
    print("background data : {}".format(background.shape))
    
    print("saving data...")
    np.save(saving_dir+"miniboone_signal.npy", signal)
    np.save(saving_dir+"miniboone_background.npy", background)


def augment_sequence_with_replacement(sequence, required_len):
    """
    augment the sequence to required_len by sampling with replacement.
    """

    shortage = required_len - sequence.shape[0]
    
    # random draw with replacement from existing f0_sequence
    sampled_idx = np.random.choice(sequence.shape[0], size=shortage, replace=True)
    extra = sequence[sampled_idx]
    
    # append sampled rows to the original sequence
    extended_sequence = np.concatenate([sequence, extra], axis=0)
    
    return extended_sequence
