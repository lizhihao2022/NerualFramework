import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os


class MyDataset:
    def __init__(
        self,
        data_dir,
        train_ratio=0.6,
        valid_ratio=0.2,
        test_ratio=0.2,
        train_batchsize=32,
        eval_batchsize=32,
        subset=False,
        sub_ratio=0.2,
        num_workers=0,
        pin_memory=False,
        ):
        self.train_batchsize = train_batchsize
        self.eval_batchsize = eval_batchsize
        X, y = self.load_data(data_dir, subset, sub_ratio)
        train_X, valid_test_X, train_y, valid_test_y = train_test_split(X, y, train_size=train_ratio)
        valid_X, test_X, valid_y, test_y = train_test_split(valid_test_X, valid_test_y, train_size=valid_ratio/(valid_ratio+test_ratio))
        
        self.train_data = MyBase(train_X, train_y)
        self.valid_data = MyBase(valid_X, valid_y)
        self.test_data = MyBase(test_X, test_y)
    
    def load_data(self, data_dir, subset, sub_ratio):
        raise NotImplementedError
    
    @property
    def train_loader(self):
        return DataLoader(self.train_data, batch_size=self.train_batchsize, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    @property
    def valid_loader(self):
        return DataLoader(self.valid_data, batch_size=self.eval_batchsize, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    @property
    def test_loader(self):
        return DataLoader(self.test_data, batch_size=self.eval_batchsize, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)


class MyBase(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
