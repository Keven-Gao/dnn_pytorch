# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 20:41:22 2021

@author: gaokaifeng
@institution: CUGB
@e-mail: kaifeng.gao@foxmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
#import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
from time import time

torch.cuda.manual_seed_all(0)

torch.set_default_tensor_type(torch.FloatTensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#define the model
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=2,out_features=32, bias=True)
        self.linear2 = torch.nn.Linear(in_features=32,out_features=64, bias=True)
        self.linear3 = torch.nn.Linear(in_features=64,out_features=128, bias=True)
        self.linear4 = torch.nn.Linear(in_features=128,out_features=128, bias=True)
        self.linear5 = torch.nn.Linear(in_features=128,out_features=128, bias=True)
        self.linear6 = torch.nn.Linear(in_features=128,out_features=64, bias=True)
        self.linear7 = torch.nn.Linear(in_features=64,out_features=32, bias=True)
        self.linear8 = torch.nn.Linear(in_features=32,out_features=1, bias=True)
    def forward(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = self.linear2(x)
        #print(x.shape)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        x = F.relu(self.linear5(x))
        x = self.linear6(x)
        x = F.relu(self.linear7(x))
        x = self.linear8(x)
        return x
'''
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=2,out_features=64, bias=True)
        self.linear2 = torch.nn.Linear(in_features=64,out_features=128, bias=True)
        self.linear3 = torch.nn.Linear(in_features=128,out_features=64, bias=True)
        self.linear4 = torch.nn.Linear(in_features=64,out_features=1, bias=True)
    def forward(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = self.linear2(x)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
'''
print("set data and model from cpu to cuda")
train = pd.read_excel(r'renormdata2_1.xlsx')
X = torch.tensor(train.loc[:,['X','Y']].values)
real_out = torch.tensor(train.loc[:,['Z']].values)
X = X.cuda()
real_out = real_out.cuda()
print(X.device)
print(real_out.device)

text = pd.read_excel(r'renormpred2_1.xlsx')
Xt = torch.tensor(text.loc[:,['X','Y']].values)
real_out_text = torch.tensor(text.loc[:,['Z']].values)
Xt = Xt.cuda()
real_out_text = real_out_text.cuda()
print(Xt.device)
print(real_out_text.device)

t1 = time()
#calculating the loss function and find the best parameters
dnn = DNN() 
dnn = dnn.float()
dnn = dnn.cuda()
print (next(dnn.parameters()).device)
mse = nn.MSELoss() 
adam = optim.Adam(lr=0.0015, params=dnn.parameters())
iteration = []
Loss = []

for i in range(30000):
    # X = X.to(device)
    # X = X.cpu()
    X = X.clone().detach().float()
    # pre_out = dnn(torch.Tensor(X)) 
    pre_out = dnn(X)
    adam.zero_grad() 
    loss1 = mse(pre_out.double(), real_out)
    loss1.backward() 
    adam.step() 
    if (i+1) % 1000==0:
        iteration.append(i)
        Loss.append(loss1)
        print(loss1)

#initialize the parameter w and b
w = torch.tensor(0.)
w.requires_grad_(True)
b = torch.tensor(0.)
b.requires_grad_(True)
lr = torch.tensor(0.01)

t2 = time()
t = t2-t1
# print(Xt.dtype)
pre_out = dnn(Xt.float())
loss = mse(pre_out.double(), real_out_text)

#此处得到的结果为MSE，为CPU版本得到的结果RMSE的平方值，计算结果一致
print("the loss is:",loss)
print("Time consumption: {0} s.".format(t))
'''
X_1=Xt[:,0]
X_2=Xt[:,1]

fig = plt.figure()
ax = Axes3D(fig)
pre_out=list(pre_out.detach().cpu().numpy().flatten())
print(pre_out)
ax.plot_trisurf(X_1, X_2, pre_out, cmap='viridis', edgecolor='none')
plt.title("train result")
plt.show()
'''


