#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 19:42:23 2018

@author: ck807
"""
import numpy as np

import tensorflow as tf

from residual import Residual

from keras.models import Model
import matplotlib.pyplot as plt

import keras
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from layers import initial_conv_block, bottleneck_block

import keras.backend as K


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
    with tf.device('/device:GPU:0'):
        inputs = Input((192, 192, 3))
    
        conv1 = initial_conv_block(inputs, weight_decay=5e-4)
        conv1 = bottleneck_block(conv1, filters=32, cardinality=32, strides=1, weight_decay=5e-4)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
        conv2 = bottleneck_block(pool1, filters=32, cardinality=32, strides=1, weight_decay=5e-4)
        conv2 = bottleneck_block(conv2, filters=64, cardinality=32, strides=1, weight_decay=5e-4)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = bottleneck_block(pool2, filters=64, cardinality=32, strides=1, weight_decay=5e-4)
        conv3 = bottleneck_block(conv3, filters=128, cardinality=32, strides=1, weight_decay=5e-4)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
        conv4 = bottleneck_block(pool3, filters=128, cardinality=32, strides=1, weight_decay=5e-4)
        conv4 = bottleneck_block(conv4, filters=256, cardinality=32, strides=1, weight_decay=5e-4)
        drop4 = Dropout(0.2)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
        conv5 = bottleneck_block(pool4, filters=256, cardinality=32, strides=1, weight_decay=5e-4)
        conv5 = bottleneck_block(conv5, filters=256, cardinality=32, strides=1, weight_decay=5e-4)
        drop5 = Dropout(0.2)(conv5)
    with tf.device('/device:GPU:1'):
        up6 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = keras.layers.Concatenate()([drop4, up6])
        conv6 = bottleneck_block(merge6, filters=128, cardinality=32, strides=1, weight_decay=5e-4)
        conv6 = bottleneck_block(conv6, filters=128, cardinality=32, strides=1, weight_decay=5e-4)
    
        up7 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = keras.layers.Concatenate()([conv3, up7])
        conv7 = bottleneck_block(merge7, filters=64, cardinality=32, strides=1, weight_decay=5e-4)
        conv7 = bottleneck_block(conv7, filters=64, cardinality=32, strides=1, weight_decay=5e-4)
    
        up8 = Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = keras.layers.Concatenate()([conv2, up8])
        conv8 = bottleneck_block(merge8, filters=32, cardinality=32, strides=1, weight_decay=5e-4)
        conv8 = bottleneck_block(conv8, filters=32, cardinality=32, strides=1, weight_decay=5e-4)
    
        up9 = Conv2D(16, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = keras.layers.Concatenate()([conv1, up9])
        conv9 = Residual(48, 16, merge9)
        conv9 = Residual(16, 4, conv9)
        conv9 = Residual(4, 1, conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
        model = Model(inputs, conv10)

        model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.03, momentum=0.9, nesterov=True), metrics=[dice_coef,jaccard_coef])

    return model

model = get_unet()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint("-val_loss_checkpoint.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6)
            ]

history = model.fit(trainData, trainMask, validation_data=(valData, valMask), batch_size=16, epochs=200, verbose=1, shuffle=True, callbacks=callbacks)

print('Saving the model..')
model.save('final_Attention.h5')

print('Saving plots..')
plt1 = plt.plot(history.history['loss'])
plt1 = plt.plot(history.history['val_loss'])
plt1 = plt.title('Loss')
plt1 = plt.xlabel('epoch')
plt1 = plt.legend(['loss', 'val_loss'], loc='upper right')
plt1 = plt.savefig('lossAutoencoder.png')

plt2 = plt.plot(history.history['dice_coef'])
plt2 = plt.plot(history.history['val_dice_coef'])
plt2 = plt.title('Dice Coef Accuracy')
plt2 = plt.xlabel('epoch')
plt2 = plt.legend(['dice_coef', 'val_dice_coef'], loc='upper left')
plt2 = plt.savefig('dice_coef_autoEncoder.png')

plt3 = plt.plot(history.history['jaccard_coef'])
plt3 = plt.plot(history.history['val_jaccard_coef'])
plt3 = plt.title('Jaccard Coef Accuracy')
plt3 = plt.xlabel('epoch')
plt3 = plt.legend(['jaccard_coef', 'val_jaccard_coef'], loc='upper left')
plt3 = plt.savefig('jaccard_coef_autoEncoder.png')

plt3 = plt.plot(history.history['lr'])
plt3 = plt.title('Learning Rate')
plt3 = plt.xlabel('epoch')
plt3 = plt.savefig('lr_autoencoder.png')
