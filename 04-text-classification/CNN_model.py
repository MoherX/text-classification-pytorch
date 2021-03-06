#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
import data_preprocess
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import os
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()

torch.set_num_threads(10)

# 将数据划分为训练集和测试集
X_train, X_test, Y_train, Y_test = data_preprocess.tensorFromData()
# trainDataSet = data_preprocess.TextDataSet(X_train, Y_train)
# testDataSet = data_preprocess.TextDataSet(X_test, Y_test)
trainDataSet = Data.TensorDataset(X_train, Y_train)
testDataSet = Data.TensorDataset(X_test, Y_test)
trainDataLoader = DataLoader(trainDataSet, batch_size=16, shuffle=True,num_workers=4)
testDataLoader = DataLoader(testDataSet, batch_size=16, shuffle=False,num_workers=4)

# 获取字典
word_to_inx, inx_to_word = data_preprocess.get_dic()
len_dic = len(word_to_inx)

# 定义超参数
MAXLEN = 64
input_dim = MAXLEN
emb_dim = 128
num_epoches = 3
batch_size = 16


# 定义模型
class CNN_model(nn.Module):
    def __init__(self, len_dic, input_dim, emb_dim):
        super(CNN_model, self).__init__()
        self.embed = nn.Embedding(len_dic, emb_dim)  # b,64,128
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),  # b,256,128
            nn.MaxPool1d(2, 2),  # b,256,64
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),  # b,128,64
            nn.MaxPool1d(2, 2),  # b,128,32
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),  # b,64,32
            nn.MaxPool1d(2, 2),  # b,64,16
        )
        self.bn = nn.BatchNorm1d(64)  # b,64,16  -> #b,64*16  same shape as input
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(64 * 16, 256)  # b,256
        self.relu = nn.ReLU(True)  # b,256
        self.classify = nn.Linear(256, 3)  # b,3

    def forward(self, x):
        x = self.embed(x)
        # print(x.size())
        # x = x.permute(0, 2, 1)  # 将通道放到第二位
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.conv3(x)
        x = self.bn(x)
        b, c, l = x.size()
        x = x.view(b, c * l)
        x = self.drop(x)
        # print(x.size())
        x = self.linear(x)
        x = self.relu(x)
        x = x.view(-1, 256)
        out = self.classify(x)
        out = F.log_softmax(out,dim=1)
        return out


if use_cuda:
    model = CNN_model(len_dic, input_dim, emb_dim).cuda()
else:
    model = CNN_model(len_dic, input_dim, emb_dim)
print(model)

criterion = nn.NLLLoss()
optimzier = optim.Adam(model.parameters(), lr=1e-3)
best_acc = 0.0
best_model = None

d1=datetime.datetime.now()

for epoch in range(num_epoches):
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for i, data in enumerate(trainDataLoader):
        x, y = data
        if use_cuda:
            x, y = Variable(x).cuda(), Variable(y).cuda()
        else:
            x, y = Variable(x), Variable(y)
        # forward
        out = model(x)
        loss = criterion(out, y)
        train_loss += loss.item() * len(y)
        _, pre = torch.max(out, 1)
        num_acc = sum(pre == y)
        train_acc += num_acc.item()
        # backward
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        if (i + 1) % 100 == 0:
            print('[{}/{}],train loss is:{:.6f},train acc is:{:.6f}'.format(i, len(trainDataLoader),
                                                                            train_loss / (i * batch_size),
                                                                            train_acc / (i * batch_size)))
    print(
        'epoch:[{}],train loss is:{:.6f},train acc is:{:.6f}'.format(epoch,
                                                                     train_loss / (len(trainDataLoader) * batch_size),
                                                                     train_acc / (len(trainDataLoader) * batch_size)))
    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    for i, data in enumerate(testDataLoader):
        x, y = data
        if use_cuda:
            x, y = Variable(x).cuda(), Variable(y).cuda()
        else:
            x, y = Variable(x), Variable(y)
        out = model(x)
        loss = criterion(out, y)
        eval_loss += loss.item() * len(y)
        _, pre = torch.max(out, 1)
        num_acc =sum(pre == y)
        eval_acc += num_acc.item()
    print('test loss is:{:.6f},test acc is:{:.6f}'.format(
        eval_loss / (len(testDataLoader) * batch_size),
        eval_acc / (len(testDataLoader) * batch_size)))
    if best_acc < (eval_acc / (len(testDataLoader) * batch_size)):
        best_acc = eval_acc / (len(testDataLoader) * batch_size)
        best_model = model.state_dict()
        print('best acc is {:.6f},best model is changed'.format(best_acc))

d2=datetime.datetime.now()
print (d2-d1).seconds

torch.save(model.state_dict(), './model/CNN_model.pth')
