from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class LungCancerDataset(Dataset):
    def __init__(self, path, split):
        self.path = path
        self.file_out = np.loadtxt(path, dtype=np.float32, delimiter=",")
        train_idx = 280

        if split == 'Train':
            self.x = self.file_out[1:train_idx, :15]
            self.y = self.file_out[1:train_idx, 15]

        elif split == 'Test':
            self.x = self.file_out[train_idx:, :15]
            self.y = self.file_out[train_idx:, 15]

        self.X_train = torch.from_numpy(self.x)/100
        self.y_train = torch.from_numpy(self.y)/100

        self.y_train = torch.tensor(self.y_train)
        # self.y_train = torch.tensor(self.y_train, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
