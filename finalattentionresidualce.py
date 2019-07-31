#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:53:10 2018

@author: ck807
"""

import numpy as np

import tensorflow as tf

from keras.models import Model
import matplotlib.pyplot as plt

import keras
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from residualblockwithlayernames import initial_conv_block1, Residual2, Residual3, Residual4, Residual5, Residual6
from residualblockwithlayernames import Residual7, Residual8, Residual9, Residual10, Residual11, Residual12
from residualblockwithlayernames import Residual13, Residual14, Residual15, Residual16, Residual17, Residual18, Residual19

import keras.backend as K


print('Loading the data..')
trainData = np.load('trainData.npy')
trainMask = np.load('trainMask.npy')
valData = np.load('valData.npy')
valMask = np.load('valMask.npy')

print('Building and compiling the model..')
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def get_unet():
    with tf.device('/device:GPU:1'):
        inputs = Input((192, 192, 3), name='Input')
        
        conv1 = initial_conv_block1(inputs)
        conv1 = Residual2(16, 32, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='MaxPool1')(conv1)
    
        conv2 = Residual3(32, 32, pool1)
        conv2 = Residual4(32, 64, conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='MaxPool2')(conv2)
    
        conv3 = Residual5(64, 64, pool2)
        conv3 = Residual6(64, 128, conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='MaxPool3')(conv3)
    
        conv4 = Residual7(128, 128, pool3)
        conv4 = Residual8(128, 256, conv4)
        drop4 = Dropout(0.2, name='Dropout1')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='MaxPool4')(drop4)
    
        conv5 = Residual9(256, 256, pool4)
        conv5 = Residual10(256, 128, conv5)
        drop5 = Dropout(0.2, name='Dropout2')(conv5)
    
        up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv1')(UpSampling2D(size = (2,2), name='Up1')(drop5))
        merge6 = keras.layers.Concatenate(name='Concat1')([drop4,up6])
        conv6 = Residual11(384, 128, merge6)
        conv6 = Residual12(128, 64, conv6)
    
        up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv2')(UpSampling2D(size = (2,2), name='Up2')(conv6))
        merge7 = keras.layers.Concatenate(name='Concat2')([conv3,up7])
        conv7 = Residual13(192, 64, merge7)
        conv7 = Residual14(64, 32, conv7)
    
        up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv3')(UpSampling2D(size = (2,2), name='Up3')(conv7))
        merge8 = keras.layers.Concatenate(name='Concat3')([conv2,up8])
        conv8 = Residual15(96, 32, merge8)
        conv8 = Residual16(32, 16, conv8)
    
        up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv4')(UpSampling2D(size = (2,2), name='Up4')(conv8))
        merge9 = keras.layers.Concatenate(name='Concat4')([conv1,up9])
        conv9 = Residual17(48, 16, merge9)
        conv9 = Residual18(16, 2, conv9)
        conv9 = Residual19(2, 1, conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid', name='Output')(conv9)
    
        model = Model(inputs, conv10)

        model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.03, momentum=0.9, nesterov=True), metrics=[dice_coef,jaccard_coef])

    return model

model = get_unet()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint("-val_loss_Residual_crossentropy_checkpoint.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6)
            ]

history = model.fit(trainData, trainMask, validation_data=(valData, valMask), batch_size=32, epochs=200, verbose=1, shuffle=True, callbacks=callbacks)

model.save('final_Attention_crossentropy_Residual.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.savefig('loss_cross_Autoencoder.png')

plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('Dice Coef Accuracy')
plt.xlabel('epoch')
plt.legend(['dice_coef', 'val_dice_coef'], loc='upper left')
plt.savefig('dice_coef_cross_autoEncoder.png')

plt.plot(history.history['jaccard_coef'])
plt.plot(history.history['val_jaccard_coef'])
plt.title('Jaccard Coef Accuracy')
plt.xlabel('epoch')
plt.legend(['jaccard_coef', 'val_jaccard_coef'], loc='upper left')
plt.savefig('jaccard_coef_cross_autoEncoder.png')

plt.plot(history.history['lr'])
plt.title('Learning Rate')
plt.xlabel('epoch')
plt.savefig('lr_cross_autoencoder.png')
