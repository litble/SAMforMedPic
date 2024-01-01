# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


class NibDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_dir = os.path.join(data_dir, 'img')
        self.label_dir = os.path.join(data_dir, 'label')
        self.img_files = os.listdir(self.img_dir)
        self.label_files = os.listdir(self.label_dir)
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_file_name = os.path.join(self.img_dir, self.img_files[idx])
        label_file_name = os.path.join(self.label_dir, self.label_files[idx])
        
        img = nib.load(img_file_name)
        label = nib.load(label_file_name)
        
        # 这里可以添加一些数据预处理步骤，如标准化、裁剪、缩放等
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
        
        return img, label

# %% sanity test of dataset class
tr_dataset = NibDataset("RawData/Training")
tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)