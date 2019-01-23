from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
# # Create dummy csv data
nb_samples = 20
# a = np.arange(nb_samples)
# df = pd.DataFrame(a, columns=['data'])
# df.to_csv('data.csv', index=False)



class CSVDataset(Dataset):
    def __init__(self, path, x, y, transform=None):
        self.path =path
        data = np.load(self.path)



        self.x = data[x]
        self.y = data[y]


        self.len = len(y)
        self.transform = transform
    def __getitem__(self, index):
        sen, label = self.x[index], self.y[index]
        scalar_list = []
        print(sen.shape)

        sen = torch.FloatTensor(sen)
        label = torch.FloatTensor(label)

        sen = sen.unsqueeze(0)

        label = label.unsqueeze(0)

        return sen, label

    def __len__(self):
        return int(self.len)


train_transform = T.Compose([
        T.ToTensor(),
    ])


trained_dataset = CSVDataset("../data/train.npz", x="X_train", y="Y_train", transform=train_transform)
test_dataset = CSVDataset("../data/test.npz", x="X_test", y="Y_test")
# print(trained_dataset[15])
train_loader = DataLoader(trained_dataset, batch_size=2, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

class SeqModel(nn.Module):
    def __init__(self):
        super(SeqModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, 3, stride=1)
        self.conv2 = nn.Conv1d(10, 3, 3, stride=1)
        self.fc1 = nn.Linear(168, 3)

    #
    def forward(self, x):
        # print(self.conv1(x))

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.fc1(x)
        return x


model = SeqModel()
model.train()

for i, data in enumerate(train_loader):
    x_train, y_train = data
    # print(data)
    # print(x_train.size())

    x_var = Variable(x_train)
    y_var = Variable(y_train)

    score = model(x_var)
    print(score)
