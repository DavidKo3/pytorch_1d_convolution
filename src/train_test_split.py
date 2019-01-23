from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split



data = pd.read_csv('../data/splice.data.txt')
x = data.iloc[:, 2].values
y = data.iloc[:, 0].values

# print(y)
# list_x = [list(str.strip()) for str in x]
list_y = [str.strip() for str in y]

# print("--"*30)
# print(list_y)
# print("--"*30)

list_x=[]
for str in x:
    # print(str.strip())
    line=[]
    for ch in str.strip():
        # print(ch, ord(ch))
        line.append(ord(ch))
    # print(line)
    list_x.append(np.array(line))

# print("-"*100)
# for i in list_x:
#     print(len(i))
# print("-"*100)



list_y=[]
for str in y:
    # print(str.strip())
    line=[]
    if str =="EI":
        # print(ch)
        line.append(0)
    elif str == "IE":
        line.append(1)
    else:
        line.append(2)
    # print(line)
    list_y.append(np.array(line))

# print(list_y)

# np_list_x = [np.array(ord(str)) for str in list_x]
np_list_x = list_x
np_list_y = [np.array(str) for str in list_y]
# print(type(np_list_x[0]))

X_train, X_test, Y_train, Y_test = train_test_split(np_list_x, np_list_y, test_size=20, random_state=2)
print("X_train-------------------------------------------------------------------------------------")
print(X_train)
print("X_train-------------------------------------------------------------------------------------")



for i in X_train:
    print(len(i))
print(len(X_train))
print(len(X_test))
#
# print(type(X_train[0]))

saved = 1

if saved ==1:
    np.savez("../data/train.npz", X_train=X_train, Y_train=Y_train)
    np.savez("../data/test.npz", X_test=X_test, Y_test=Y_test)

train_data = np.load("../data/train.npz")
test_data = np.load("../data/test.npz")

# print(type(train_data["Y_train"][0]))
# print(train_data["Y_train"][0])

for i, data in enumerate(train_data["X_train"]):
    print("----------------------------------\n")
    print(data)
    print(X_train[i])
    print(len(data))
    print("----------------------------------\n")