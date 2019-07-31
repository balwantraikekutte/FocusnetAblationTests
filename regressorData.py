#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:05:33 2018

@author: ck807
"""

import os, glob
import numpy as np
import pandas as pd
import cv2


i = 0
data_file = glob.glob('/local/data/chaitanya/landmarker/images/train/*.png')
files = []

data_file_label = glob.glob('/local/data/chaitanya/landmarker/txt/train/*.txt')

trainData = np.zeros((len(data_file),192, 192, 3))
trainLabel = np.zeros((len(data_file_label), 20))

print('Generating training set..')
for f in (data_file):
    a=cv2.imread(f)
    trainData[i,:,:,:] = a[:,:,:]
    base = os.path.basename("/local/data/chaitanya/landmarker/images/train/" + f)
    fileName = os.path.splitext(base)[0]
    files.append(fileName)
    i += 1

print('Generating training set labels..')
for k in (data_file_label):
    base = os.path.basename("/local/data/chaitanya/landmarker/txt/train/" + k)
    fileName = os.path.splitext(base)[0]
    fileName = fileName + '_depth'
    index = files.index(fileName)
    txt_file = pd.read_csv(k)
    txt_file = txt_file.as_matrix()
    txt_file = txt_file.ravel()
    trainLabel[index, :] = txt_file[:]
    
i = 0
data_file_val = glob.glob('/local/data/chaitanya/landmarker/images/val/*.png')
files_val = []

data_file_label_val = glob.glob('/local/data/chaitanya/landmarker/txt/val/*.txt')

valData = np.zeros((len(data_file_val),192, 192, 3))

valLabel = np.zeros((len(data_file_label_val), 20))

print('Generating validation set..')
for f in (data_file_val):
    a=cv2.imread(f)
    valData[i,:,:,:] = a[:,:,:]
    base = os.path.basename("/local/data/chaitanya/landmarker/images/val/" + f)
    fileName = os.path.splitext(base)[0]
    files_val.append(fileName)
    i += 1

print('Generating validation set labels..')
for k in (data_file_label_val):
    base = os.path.basename("/local/data/chaitanya/landmarker/txt/val/" + k)
    fileName = os.path.splitext(base)[0]
    fileName = fileName + '_depth'
    index = files_val.index(fileName)
    txt_file = pd.read_csv(k)
    txt_file = txt_file.as_matrix()
    txt_file = txt_file.ravel()
    valLabel[index, :] = txt_file[:,]

print('PreProcessing the data..')
trainData = trainData.astype('float32')
trainDataMean = np.mean(trainData)
trainDataStd = np.std(trainData)

trainData -= trainDataMean
trainData /= trainDataStd

trainLabel = trainLabel.astype('float32')

valData = valData.astype('float32')

valData -= trainDataMean
valData /= trainDataStd

valLabel = valLabel.astype('float32')

print('Saving as npy files..')
np.save('trainDataRegressor.npy',trainData)
np.save('trainLabelRegressor.npy', trainLabel)
np.save('valDataRegressor.npy',valData)
np.save('valLabelRegressor.npy', valLabel)
