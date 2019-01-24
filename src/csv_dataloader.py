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
import torchvision.models as models
import torch.optim as optim



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


        self.len = len(self.y)
        self.transform = transform
    def __getitem__(self, index):
        sen, label = self.x[index], self.y[index]

        sen = torch.FloatTensor(sen)
        label = torch.LongTensor(label)

        sen = sen.unsqueeze(0)

        return sen, label

    def __len__(self):
        return self.len


batch_size= 10

train_transform = T.Compose([
        T.ToTensor(),
    ])


trained_dataset = CSVDataset("../data/train.npz", x="X_train", y="Y_train", transform=train_transform)
test_dataset = CSVDataset("../data/test.npz", x="X_test", y="Y_test")
# print(trained_dataset[15])
train_loader = DataLoader(trained_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("len of train_loader : ", len(train_loader))
print("len of test_loader : ", len(test_loader))
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
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc1(x)
        return x



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



model = SeqModel()
model.to(device)
model.train()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


epochs=100
for epoch in range(epochs):
    print('\n-----> epoch %d ' % epoch)
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        print(i, data)
        x_train, y_train = data
        # print(data)
        # print(x_train.size())
        # print("x_train :", x_train)
        # print("y_train :", y_train)

        x_var = Variable(x_train.cuda())
        y_var = Variable(y_train.cuda())
        # print("x_var :", x_var)
        # print("y_var :", y_var)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        outputs = model(x_var)

        y_var = y_var.squeeze()

        if y_var.dim() ==0:
            y_var = torch.cuda.LongTensor([y_var])
        loss = criterion(outputs, y_var)


        loss.backward()
        optimizer.step()

        # print statisitics
        running_loss += loss.item()

        if i % batch_size == 0:
            print(" [%d, %5d] loss : %.3f" % (epoch+1, i+1, running_loss/batch_size))
            running_loss=0.0








