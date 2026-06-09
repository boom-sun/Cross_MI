#!/usr/bin/env python
# coding: utf-8
"""
All network architectures: FBCNet, EEGNet, DeepConvNet
@author: Ravikiran Mane
"""
import torch
import torch.nn as nn
import sys
import numpy as np
import scipy.signal
from .base import SkorchNet
# from smt import SMT

current_module = sys.modules[__name__]

debug = False


# %% Deep convnet - Baseline 1
# class deepConvNet(nn.Module):
#     def convBlock(self, inF, outF, dropoutP, kernalSize, *args, **kwargs):
#         return nn.Sequential(
#             nn.Dropout(p=dropoutP),
#             Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
#             nn.BatchNorm2d(outF),
#             nn.ELU(),
#             nn.MaxPool2d((1, 3), stride=(1, 2))  # stride = (1,3)
#         )
#
#     def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
#         return nn.Sequential(
#             Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
#             Conv2dWithConstraint(25, 25, (nChan, 1), padding=0, bias=False, max_norm=2),
#             nn.BatchNorm2d(outF),  # 归一化
#             nn.ELU(),  # 激活函数
#             nn.MaxPool2d((1, 3), stride=(1, 3))
#         )
#
#     def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
#         return nn.Sequential(
#             Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs),
#             nn.LogSoftmax(dim=1))
#
#     def calculateOutSize(self, model, nChan, nTime):
#         '''
#         Calculate the output based on input size.
#         model is from nn.Module and inputSize is a array.
#         '''
#         data = torch.rand(1, 1, nChan, nTime)
#         model.eval()
#         out = model(data).shape
#         return out[2:]
#
#     def __init__(self, nChan, nTime, nClass=2, dropoutP=0.25, *args, **kwargs):
#         super(deepConvNet, self).__init__()
#
#         kernalSize = (1, 10)
#         nFilt_FirstLayer = 25
#         nFiltLaterLayer = [25, 50, 100, 200]
#
#         firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
#         middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
#                                        for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])
#
#         self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)
#
#         self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
#         self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))
#
#     def forward(self, x):
#         x = self.allButLastLayers(x)
#         x = self.lastLayer(x)
#         x = torch.squeeze(x, 3)
#         x = torch.squeeze(x, 2)
#
#         return x


# %% EEGNet Baseline 2
# class eegNet(nn.Module):
#     def initialBlocks(self, dropoutP, *args, **kwargs):
#         block1 = nn.Sequential(
#             nn.Conv2d(1, self.F1, (1, self.C1),
#                       padding=(0, self.C1 // 2), bias=False),
#             nn.BatchNorm2d(self.F1),
#             Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
#                                  padding=0, bias=False, max_norm=1,
#                                  groups=self.F1),
#             nn.BatchNorm2d(self.F1 * self.D),
#             nn.ELU(),
#             nn.AvgPool2d((1, 4), stride=4),
#             nn.Dropout(p=dropoutP))
#         block2 = nn.Sequential(
#             nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 22),
#                       padding=(0, 22 // 2), bias=False,
#                       groups=self.F1 * self.D),
#             nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
#                       stride=1, bias=False, padding=0),
#             nn.BatchNorm2d(self.F2),
#             nn.ELU(),
#             nn.AvgPool2d((1, 8), stride=8),
#             nn.Dropout(p=dropoutP)
#         )
#         return nn.Sequential(block1, block2)
#
#     def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
#         return nn.Sequential(
#             nn.Conv2d(inF, outF, kernalSize, *args, **kwargs),
#             nn.LogSoftmax(dim=1))
#
#     def calculateOutSize(self, model, nChan, nTime):
#         '''
#         Calculate the output based on input size.
#         model is from nn.Module and inputSize is a array.
#         '''
#         data = torch.rand(1, 1, nChan, nTime)
#         model.eval()
#         out = model(data).shape
#         return out[2:]
#
#     def __init__(self, nChan, nTime, nClass=2,
#                  dropoutP=0.25, F1=8, D=2,
#                  C1=125, *args, **kwargs):
#         super(eegNet, self).__init__()
#         self.F2 = D * F1
#         self.F1 = F1
#         self.D = D
#         self.nTime = nTime
#         self.nClass = nClass
#         self.nChan = nChan
#         self.C1 = C1
#
#         self.firstBlocks = self.initialBlocks(dropoutP)
#         self.fSize = self.calculateOutSize(self.firstBlocks, self.nChan, self.nTime)
#         self.lastLayer = self.lastBlock(self.F2, self.nClass, (1, self.fSize[1]))
#
#     def forward(self, x):
#         # print(x.shape)
#         x = self.firstBlocks(x)
#         # print(x.shape)
#         x = self.lastLayer(x)
#         # print(x.shape)
#         x = torch.squeeze(x, 3)
#         x = torch.squeeze(x, 2)
#
#         return x


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


# %% Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''

    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim=self.dim, keepdim=True)


class StdLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''

    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim=self.dim, keepdim=True)


class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''

    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''

    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=True)


class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''

    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma, ima = x.max(dim=self.dim, keepdim=True)
        return ma


class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''

    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data, axis=-1)
    return y

def generate_filter_bands(data, fs, bands):
    batch_size, n_channels, n_samples = data.shape
    n_bands = len(bands)
    filtered_data = np.zeros((batch_size, n_channels, n_samples, n_bands))
    for i, (low, high) in enumerate(bands):
        for b in range(batch_size):
            for c in range(n_channels):
                filtered_data[b, c, :, i] = bandpass_filter(data[b, c, :], low, high, fs)
    return filtered_data

@SkorchNet
class FBCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    '''
        FBNet with seperate variance for every 1s. 
        The data input is in a form of batch x 1 x chan x time x filterBand
    '''

    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
            Conv2dWithConstraint(nBands, m * nBands, (nChan, 1), groups=nBands,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(m * nBands),
            swish()
        )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))

    def __init__(self, nChan, nTime, nClass=2, nBands=9, m=32,
                 temporalLayer='LogVarLayer', strideFactor=4, doWeightNorm=True, *args, **kwargs):
        super().__init__()  ###############7.26

        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        # create all the parrallel SCBc
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm=doWeightNorm)

        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)

        # The final fully connected layer
        self.lastLayer = self.LastBlock(self.m * self.nBands * self.strideFactor, nClass, doWeightNorm=doWeightNorm)

    def forward(self, x):
        # 调整输入的维度 [batch, channels, time] -> [batch, channels, time, filterBands]
        x_filtered = generate_filter_bands(x.numpy(), fs=250, bands = [(0.5, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70)])
        x_filtered = torch.tensor(x_filtered, dtype=torch.double).unsqueeze(1)

        # 进行空间卷积
        x_filtered = x_filtered.permute(0, 4, 2, 3, 1)  # 调整维度顺序为 [batch, filterBands, channels, time, 1]
        x_filtered = x_filtered.squeeze(-1)  # 移除最后一个多余的维度

        x = self.scb(x_filtered)

        # 调整形状以适应时间层的输入
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])

        # 通过时间层
        x = self.temporalLayer(x)

        # 扁平化输出
        x = torch.flatten(x, start_dim=1)

        # 最后的全连接层输出
        x = self.lastLayer(x)

        return x



