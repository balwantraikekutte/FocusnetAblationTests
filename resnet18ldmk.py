#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 20:30:32 2018

@author: ck807
"""

from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import keras.backend as K
from keras.optimizers import SGD, Adam
import numpy as np
import resnet

w = 10.0
e = 2.0
c = w - w * K.log(1 + (w/e))

def wingLoss(y_true, y_pred, w=w, e=e, c=c):
    error = y_true - y_pred
    cond = K.abs(error) < w
    true = w * (K.log(1 + (K.abs(error)/e)))
    otherwise = K.abs(error) - c
    return tf.where(cond, true, otherwise)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, mode='auto', epsilon=0.001, cooldown=0, min_lr=0.5e-7)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)

batch_size = 32
nb_classes = 20
nb_epoch = 200

trainData = np.load('trainDataRegressor.npy')
trainLabel = np.load('trainLabelRegressor.npy')
valData = np.load('valDataRegressor.npy')
valLabel = np.load('valLabelRegressor.npy')

model = resnet.ResnetBuilder.build_resnet_18((3, 192, 192), nb_classes)
model.summary()
model.compile(loss=wingLoss,
              optimizer='Adam',
              metrics=['mae'])

model.fit(trainData, trainLabel,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(valData, valLabel),
          shuffle=True,
          callbacks=[lr_reducer, early_stopper])

#SGD(lr=1e-1, momentum=0.9, nesterov=True)
