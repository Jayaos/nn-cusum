import numpy as np
import pandas as pd
import arff
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


def susy_data_processing(data_dir, saving_dir, low_feature=True):
    # refer https://github.com/alyashgo/SUSY-Dataset--EDA-Classification/blob/master/Susy.ipynb for data description
    susy = pd.read_csv(data_dir, header=None)
    
    standardscaler = StandardScaler()
    standardized_data = standardscaler.fit_transform(susy.iloc[:,1:]) # the first column is label
    signal_idx = np.array(susy[0]).nonzero()[0]
    bignal_idx = np.argwhere(np.array(susy[0]) == 0).flatten()
    
    if low_feature: 
        # the first 8 features are low features, rest of the features are functions of the low features.
        signal = standardized_data[signal_idx, :8]
        background = standardized_data[bignal_idx, :8]
    else:
        signal = standardized_data[signal_idx, :]
        background = standardized_data[bignal_idx, :]

    print("signal shape: {}".format(signal.shape))
    print("background shape: {}".format(background.shape))
        
    print("saving data...")
    if low_feature:
        np.save(saving_dir+"susy_low_signal.npy", signal)
        np.save(saving_dir+"susy_low_background.npy", background)
    else:
        np.save(saving_dir+"susy_all_signal.npy", signal)
        np.save(saving_dir+"susy_all_background.npy", background)


def codrna_data_processing(data_dir, saving_dir):
    """
    Process cod-rna ARFF dataset and save two numpy arrays:
      - codrna_positive.npy
      - codrna_negative.npy

    Args:
        data_dir (str): path to .arff file
        saving_dir (str): directory prefix to save npy files
        positive_label: which label should be treated as positive class.
                        Usually 1 or '1'. The other class is treated as negative.
    """

    # load arff
    with open(data_dir, "r") as f:
        dataset = arff.load(f)

    attributes = dataset["attributes"]
    data = dataset["data"]
    columns = [attr[0] for attr in attributes]
    n_cols = len(columns)

    # convert sparse/dense rows into dense matrix
    rows = []
    for row in data:
        if isinstance(row, dict):
            dense = [0] * n_cols
            for k, v in row.items():
                dense[int(k)] = v
            rows.append(dense)
        else:
            rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

    # decode bytes if needed
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

    # assume last column is label
    label_col = df.columns[0]
    feature_cols = df.columns[1:]

    # convert features to numeric
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    # convert labels
    y_raw = df[label_col].astype(str).str.strip()

    # normalize label representation
    def _normalize_label(val):
        if val in ["1", "+1", "1.0"]:
            return 1
        elif val in ["-1", "-1.0", "0"]:
            return -1
        else:
            raise ValueError(f"Unexpected label value: {val}")

    y = y_raw.apply(_normalize_label).to_numpy()

    # standardize features
    standardscaler = StandardScaler()
    X_scaled = standardscaler.fit_transform(X).astype(np.float32)

    # choose positive vs negative
    pos_value = 1
    neg_value = -1

    positive = X_scaled[y == pos_value]
    negative = X_scaled[y == neg_value]

    print(f"label column: {label_col}")
    print(f"positive shape: {positive.shape}")
    print(f"negative shape: {negative.shape}")

    print("saving data...")
    np.save(saving_dir + "codrna_positive.npy", positive)
    np.save(saving_dir + "codrna_negative.npy", negative)


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
