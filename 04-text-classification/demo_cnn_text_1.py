#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#

"""
File: demo_cnn_text_1.py
Author: zhangxinzhan(zhangxinzhan@baidu.com)
Date: 2018/5/25 下午5:38
"""
#http://www.aclweb.org/anthology/D14-1181

import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import data_preprocess


X_train, X_test, Y_train, Y_test = data_preprocess.tensorFromData()
trainDataSet = data_preprocess.TextDataSet(X_train, Y_train)
testDataSet = data_preprocess.TextDataSet(X_test, Y_test)
trainDataLoader = DataLoader(trainDataSet, batch_size=16, shuffle=True)
testDataLoader = DataLoader(testDataSet, batch_size=16, shuffle=False)

# 获取字典
word_to_inx, inx_to_word = data_preprocess.get_dic()
len_dic = len(word_to_inx)


MAXLEN = 64
input_dim = MAXLEN
emb_dim = 128
num_epoches = 20
batch_size = 16


#定义模型
# https://pytorch.org/docs/stable/nn.html?highlight=maxpool1d#maxpool1d Input: (N,C,L^in) Output: (N,C,L^out)
class TextCNN_model(nn.Module):
    def __init__(self,len_dic,emb_dim,input_dim):
        super(TextCNN_model,self).__init__()
        self.embed=nn.Embedding(len_dic,emb_dim)#b,64,128
        self.conv1=nn.Sequential(
            nn.Conv1d(input_dim,256,1,1,padding=0),#b,256,128
            nn.ReLU(True),
            nn.MaxPool1d(2,2)#b,256,64
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(input_dim,256,3,1,padding=1),#b,256,128
            nn.ReLU(True),
            nn.MaxPool1d(2,2)#b,256,64
        )
        self.conv3=nn.Sequential(
            nn.Conv1d(input_dim,256,5,1,padding=2),#b,256,128
            nn.ReLU(True),
            nn.MaxPool1d(2,2)#b,256,64
        )
        #b,256,64+64+64
        #b,256*192
        self.drop=nn.Dropout(0.2)#b,256*192
        self.classify=nn.Linear(256*192,3)#b,3
    def forward(self, x):
        x=self.embed(x)
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.conv3(x)
        # torch.cat()
        x=torch.cat((x1,x2,x3),2) #b,256,192=64+64+64
        b,c,d=x.size() #b,256,192
        x=x.view(-1,c*d)
        x=self.drop(x)
        out=self.classify(x)
        return out


model=TextCNN_model(len_dic,emb_dim,input_dim)
print model
for i,data in enumerate(trainDataLoader):
    x,y=data

    print x,x.size()
    print x[1],x[1].size()
    # z= model.embed(x)
    print y
    # print z.size()
    break