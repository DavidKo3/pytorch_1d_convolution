from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

# # Create dummy csv data
nb_samples = 20
# a = np.arange(nb_samples)
# df = pd.DataFrame(a, columns=['data'])
# df.to_csv('data.csv', index=False)



class CSVDataset(Dataset):
    def __init__(self, path):
        self.path =path

        data = pd.read_csv(self.path)

        x = data.iloc[:, 2].values
        y = data.iloc[:, 0].values


        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=20, random_state=2)

        self.x_train= X_train
        self.x_test = X_test
        self.y_train = Y_train
        self.y_test = Y_test

        self.len = len(y)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index], self.x_test[index], self.y_test[index]

    def __len__(self):
        return int(self.len)



dataset = CSVDataset('../data/splice.data.txt')
print(dataset.len)
print(dataset.x_train[1])
print(dataset.y_train[1])

print(dataset.x_test[1])
print(dataset.y_test[1])









