#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 19:46:03 2018

@author: ck807
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.models import Model

from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dropout, BatchNormalization, Activation, Dense, GlobalAveragePooling2D, Flatten
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print('Loading the data..')
trainData = np.load('trainDataRegressor.npy')
trainLabel = np.load('trainLabelRegressor.npy')
valData = np.load('valDataRegressor.npy')
valLabel = np.load('valLabelRegressor.npy')

noutput = 20
image_dim = 192

print('Building and compiling the model..')
with tf.device('/device:GPU:2'):
    inputs = Input(shape=(image_dim, image_dim, 3))

    rconv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', name='rconv1_1')(inputs)
    rconv1 = BatchNormalization(name='rBN1_1')(rconv1)
    rconv1 = Activation('relu', name ='rActivation1_1')(rconv1)
    rpool1 = MaxPooling2D(pool_size=(2, 2), name='rMaxPool1')(rconv1)

    rconv2 = Conv2D(96, (3,3), padding='same', kernel_initializer='he_normal', name='rconv2_1')(rpool1)
    rconv2 = BatchNormalization(name='rBN2_1')(rconv2)
    rconv2 = Activation('relu', name ='rActivation2_1')(rconv2)
    rpool2 = MaxPooling2D(pool_size=(2, 2), name='rMaxPool2')(rconv2)
    
    rconv3 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', name='rconv3_1')(rpool2)
    rconv3 = BatchNormalization(name='rBN3_1')(rconv3)
    rconv3 = Activation('relu', name ='rActivation3_1')(rconv3)
    rpool3 = MaxPooling2D(pool_size=(2, 2), name='rMaxPool3')(rconv3)
    
    rconv4 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', name='rconv4_1')(rpool3)
    rconv4 = BatchNormalization(name='rBN4_1')(rconv4)
    rconv4 = Activation('relu', name ='rActivation4_1')(rconv4)
    rpool4 = MaxPooling2D(pool_size=(2, 2), name='rMaxPool4')(rconv4)
    
    rconv5 = Conv2D(384, (3,3), padding='same', kernel_initializer='he_normal', name='rconv5_1')(rpool4)
    rconv5 = BatchNormalization(name='rBN5_1')(rconv5)
    rconv5 = Activation('relu', name ='rActivation5_1')(rconv5)
    rpool5 = MaxPooling2D(pool_size=(2, 2), name='rMaxPool5')(rconv5)

    x = Flatten(name='flatten')(rpool5)
    x = Dense(128, use_bias=False, kernel_regularizer=l2(5e-4), kernel_initializer='he_normal', activation='relu', name='dense1')(x)
    x = Dropout(0.5, name='rDropout1')(x)

    output = Dense(noutput, use_bias=False, kernel_regularizer=l2(5e-4), kernel_initializer='he_normal', activation='linear', name='out')(x)

    model = Model(inputs, output)

    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['mae'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint("-val_loss_checkpoint-regressorwithoutattention.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6)
            ]

history = model.fit(trainData, trainLabel, validation_data=(valData, valLabel), batch_size=32, epochs=200, verbose=1, shuffle=True, callbacks=callbacks)

model.save('convregressorwithoutattention.h5')

plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.savefig('lossregressorwithoutattentionvonv.png')

plt.figure(1)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Mean Absolute Error Accuracy')
plt.xlabel('epoch')
plt.legend(['mae', 'val_mae'], loc='upper right')
plt.savefig('mae_regressorwithoutattentionconv.png')

plt.figure(2)
plt.plot(history.history['lr'])
plt.title('Learning Rate')
plt.xlabel('epoch')
plt.savefig('lr_regressorwithoutattentionconv.png')
