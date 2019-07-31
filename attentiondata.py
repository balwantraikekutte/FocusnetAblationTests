#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 01:25:40 2018

@author: ck807
"""

import os, glob
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

########################################################################################################################
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

print('Creating Expression Set')

i = 0
data_file = glob.glob('/local/data/chaitanya/attentionExp/images/train/*.png')
files = []

data_file_mask = glob.glob('/local/data/chaitanya/attentionExp/binaryLdmkImages/train9ldmk/*.png')

trainData1 = np.zeros((len(data_file),192, 192, 3))

trainLabel1 = np.zeros((len(data_file_mask), 192, 192, 1))

for f in (data_file):
    a=cv2.imread(f)
    trainData1[i,:,:,:] = a[:,:,:]
    base = os.path.basename("/local/data/chaitanya/attentionExp/images/train/" + f)
    fileName = os.path.splitext(base)[0]
    files.append(fileName)
    i += 1
    
for k in (data_file_mask):
    base = os.path.basename("/local/data/chaitanya/attentionExp/binaryLdmkImages/train9ldmk/" + k)
    fileName = os.path.splitext(base)[0]
    fileName = fileName + '_depth'
    index = files.index(fileName)
    image = cv2.imread(k)
    gray = rgb2gray(image)
    gray_image = img_to_array(gray)
    trainLabel1[index, :, :, :] = gray_image[:, :, :]
    
i = 0
data_file_val = glob.glob('/local/data/chaitanya/attentionExp/images/val/*.png')
files_val = []

data_file_mask_val = glob.glob('/local/data/chaitanya/attentionExp/binaryLdmkImages/val9ldmk/*.png')

valData1 = np.zeros((len(data_file_val),192, 192, 3))

valLabel1 = np.zeros((len(data_file_mask_val), 192, 192, 1))

for f in (data_file_val):
    a=cv2.imread(f)
    valData1[i,:,:,:] = a[:,:,:]
    base = os.path.basename("/local/data/chaitanya/attentionExp/images/val/" + f)
    fileName = os.path.splitext(base)[0]
    files_val.append(fileName)
    i += 1
    
for k in (data_file_mask_val):
    base = os.path.basename("/local/data/chaitanya/attentionExp/binaryLdmkImages/val9ldmk/" + k)
    fileName = os.path.splitext(base)[0]
    fileName = fileName + '_depth'
    index = files_val.index(fileName)
    image = cv2.imread(k)
    gray = rgb2gray(image)
    gray_image = img_to_array(gray)
    valLabel1[index, :, :, :] = gray_image[:, :, :]

########################################################################################################################

print('Creating Neutral Set')

i = 0
data_filen = glob.glob('/local/data/chaitanya/attentionNeutral/images/train/*.png')
filesn = []

data_file_maskn = glob.glob('/local/data/chaitanya/attentionNeutral/binaryLdmkImages/train9ldmk/*.png')

trainData2 = np.zeros((len(data_filen),192, 192, 3))

trainLabel2 = np.zeros((len(data_file_maskn), 192, 192, 1))

for f in (data_filen):
    a=cv2.imread(f)
    trainData2[i,:,:,:] = a[:,:,:]
    base = os.path.basename("/local/data/chaitanya/attentionNeutral/images/train/" + f)
    fileName = os.path.splitext(base)[0]
    filesn.append(fileName)
    i += 1
    
for k in (data_file_maskn):
    base = os.path.basename("/local/data/chaitanya/attentionNeutral/binaryLdmkImages/train9ldmk/" + k)
    fileName = os.path.splitext(base)[0]
    fileName = fileName + '_depth'
    index = filesn.index(fileName)
    image = cv2.imread(k)
    gray = rgb2gray(image)
    gray_image = img_to_array(gray)
    trainLabel2[index, :, :, :] = gray_image[:, :, :]
    
i = 0
data_file_valn = glob.glob('/local/data/chaitanya/attentionNeutral/images/val/*.png')
files_valn = []

data_file_mask_valn = glob.glob('/local/data/chaitanya/attentionNeutral/binaryLdmkImages/val9ldmk/*.png')

valData2 = np.zeros((len(data_file_valn),192, 192, 3))

valLabel2 = np.zeros((len(data_file_mask_valn), 192, 192, 1))

for f in (data_file_valn):
    a=cv2.imread(f)
    valData2[i,:,:,:] = a[:,:,:]
    base = os.path.basename("/local/data/chaitanya/attentionNeutral/images/val/" + f)
    fileName = os.path.splitext(base)[0]
    files_valn.append(fileName)
    i += 1
    
for k in (data_file_mask_valn):
    base = os.path.basename("/local/data/chaitanya/attentionNeutral/binaryLdmkImages/val9ldmk/" + k)
    fileName = os.path.splitext(base)[0]
    fileName = fileName + '_depth'
    index = files_valn.index(fileName)
    image = cv2.imread(k)
    gray = rgb2gray(image)
    gray_image = img_to_array(gray)
    valLabel2[index, :, :, :] = gray_image[:, :, :]

########################################################################################################################

print('Creating Concatenated Set')

trainData = np.zeros((16000 ,192, 192, 3))
trainData[0:8000, :, :, :] = trainData1[:, :, :, :]
trainData[8000:16000, :, :, :] = trainData2[:, :, :, :]

trainLabel = np.zeros((16000 ,192, 192, 1))
trainLabel[0:8000, :, :, :] = trainLabel1[:, :, :, :]
trainLabel[8000:16000, :, :, :] = trainLabel2[:, :, :, :]

valData = np.zeros((4000 ,192, 192, 3))
valData[0:2000, :, :, :] = valData1[:, :, :, :]
valData[2000:4000, :, :, :] = valData2[:, :, :, :]

valLabel = np.zeros((4000 ,192, 192, 1))
valLabel[0:2000, :, :, :] = valLabel1[:, :, :, :]
valLabel[2000:4000, :, :, :] = valLabel2[:, :, :, :]

########################################################################################################################

print('Preprocessing and saving tensors')

trainData = trainData.astype('float32')
trainDataMean = np.mean(trainData)
trainDataStd = np.std(trainData)

trainData -= trainDataMean
trainData /= trainDataStd

trainLabel = trainLabel.astype('float32')
trainLabel /= 255.

valData = valData.astype('float32')

valData -= trainDataMean
valData /= trainDataStd

valLabel = valLabel.astype('float32')
valLabel /= 255.
    
np.save('train9DataAttention.npy',trainData)
np.save('train9MaskAttention.npy', trainLabel)
np.save('val9DataAttention.npy',valData)
np.save('val9MaskAttention.npy', valLabel)

########################################################################################################################
