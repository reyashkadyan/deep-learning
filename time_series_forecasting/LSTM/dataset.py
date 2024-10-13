import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, target, seq_length):

    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = target[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def prepare_train_data(df, feat_cols, target_col, feat_scaler=MinMaxScaler(), target_scaler=MinMaxScaler(), seq_length=5):

    feat_data = df[feat_cols].values
    target_data = df[target_col].values
    feat_data_scaled = feat_scaler.fit_transform(feat_data)
    target_data_scaled = target_scaler.fit_transform(target_data.reshape(-1, 1))

    X, y = create_sequences(feat_data_scaled, target_data_scaled, seq_length)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor, feat_scaler, target_scaler

def prepare_test_data(df, feat_cols, target_col, feat_scaler, target_scaler, seq_length=5, scale_target=False):

    feat_data = df[feat_cols].values
    target_data = df[target_col].values
    feat_data_scaled = feat_scaler.transform(feat_data)
    target_data_scaled = target_scaler.transform(target_data.reshape(-1, 1))

    
    if not scale_target:
        X, y = create_sequences(feat_data_scaled, target_data, seq_length)
    else:
        print('Scaling features and labels...')
        X, y = create_sequences(feat_data_scaled, target_data_scaled, seq_length)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor