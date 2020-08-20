import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import os
import sys

class MyDataset(Dataset):
    def __init__(self, data_dir, train=True):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
