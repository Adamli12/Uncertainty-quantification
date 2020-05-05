import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import time


'''
Densenet
Obersevations: https://archive.org/details/github.com-liuzhuang13-DenseNet_-_2017-07-23_18-42-00
-Wide-DenseNet-BC (L=40, k=36) uses less memory/time while achieves about the same accuracy as DenseNet-BC (L=100, k=12).
-Wide-DenseNet-BC (L=40, k=48) uses about the same memory/time as DenseNet-BC (L=100, k=12), while is much more accurate.
'''

def MC_dropout(act_vec, p, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=True)

class DenseBlock(nn.Module):
    def __init__(self,in_channels,bottleneck,growth_rate,dropout_prob,n_blocks):
        super().__init__()
        self.layers=nn.ModuleList()
        self.n_blocks=n_blocks
        for i in range(n_blocks):
            self.layers.append(DenseBlockLayer(in_channels+i*growth_rate,bottleneck,growth_rate,dropout_prob))
        
    def forward(self, in_tensor):
        for i in range(self.n_blocks):
            in_tensor=self.layers[i](in_tensor)
        return in_tensor

class DenseBlockLayer(nn.Module):
    def __init__(self,in_channels,bottleneck,growth_rate,dropout_prob):
        super().__init__()
        self.dropout_prob=dropout_prob
        self.bn0=nn.modules.BatchNorm2d(in_channels)
        self.bn1=nn.modules.BatchNorm2d(bottleneck*growth_rate)
        self.conv0=nn.modules.Conv2d(in_channels,bottleneck*growth_rate,kernel_size=1,stride=1,bias=False)
        self.conv1=nn.modules.Conv2d(bottleneck*growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=False)

    def forward(self, in_tensor):
        x = self.bn0(in_tensor)
        x = F.relu(x)
        x = self.conv0(x)
        x = MC_dropout(x,self.dropout_prob)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = MC_dropout(x,self.dropout_prob)

        in_tensor = torch.cat([in_tensor,x],1)

        return in_tensor

class Transition(nn.Module):
    def __init__(self,in_channels,theta):
        super().__init__()
        self.norm=nn.modules.BatchNorm2d(in_channels)
        self.conv=nn.modules.Conv2d(in_channels,int(theta*in_channels),kernel_size=1,stride=1,bias=False)
        self.pool=nn.modules.AvgPool2d(kernel_size=2, stride=2)

    def forward(self,in_tensor):
        x=self.norm(in_tensor)
        x=F.relu(x)
        x=self.conv(x)
        x=self.pool(x)
        return x

class DenseNet(torch.nn.Module):
    """
    Densenet & MCDropout
    """

    def __init__(self,dropout_prob=0.5,growth_rate=12,bottleneck=4,init_channels=24,trans_ratio=0.5,n_classes=10,depth=40,n_dense_blocks=3,n_var=10):
        super().__init__()
        self.name="DenseNet"
        self.growth_rate = growth_rate # 12
        self.n_classes = n_classes # 10
        self.n_dense_blocks = n_dense_blocks
        self.n_var = n_var

        'Depth must be (n_dense_blocks)N + 4'
        assert (depth - 4) % n_dense_blocks == 0
        layers_per_block = int((depth - 4) / n_dense_blocks) // 2 # when n_dense_blocks=3, 6 for L=40, 16 for L=100
        self.blocks = [layers_per_block for _ in range(n_dense_blocks)]
        self.dropout_prob=dropout_prob

        self.conv=nn.modules.Conv2d(3,init_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.norm=nn.modules.BatchNorm2d(init_channels)
        self.pool=nn.modules.AvgPool2d(kernel_size=2, stride=2)
        self.denses=nn.modules.ModuleList()
        self.transes=nn.modules.ModuleList()
        num_channels=init_channels
        for i in range(n_dense_blocks):
            self.denses.append(DenseBlock(num_channels,bottleneck,growth_rate,dropout_prob,layers_per_block))
            num_channels+=layers_per_block*growth_rate
            if i!=n_dense_blocks-1:
                self.transes.append(Transition(num_channels,trans_ratio))
                num_channels=int(num_channels*trans_ratio)
        self.norm1=nn.modules.BatchNorm2d(num_channels)
        self.pool1=nn.modules.AvgPool2d(8)
        self.classifier_linear=nn.modules.Linear(num_channels,n_classes)
        self.logvar_linear=nn.modules.Linear(num_channels,n_var)

    def forward(self, input):
        x=self.conv(input)
        """
        x=self.norm(x)
        x=F.relu(x)
        x=self.pool(x)
        """
        for i in range(self.n_dense_blocks):
            x=self.denses[i](x)
            if i!=self.n_dense_blocks-1: # no transition after last denseblock
                x=self.transes[i](x)
        # no BN-RELU if using using squeeze excitation block
        x=self.norm1(x)
        x=F.relu(x)
        x=self.pool1(x).squeeze()
        y_hat=self.classifier_linear(x)

        #------predict input noise------
        log_var=self.logvar_linear(x)
        y_hat = torch.cat([y_hat, log_var],1)
        #-------------------------------

        return y_hat