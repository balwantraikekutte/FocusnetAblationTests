#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 14:20:41 2018

@author: ck807
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.models import Model

import keras
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dropout, BatchNormalization, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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
    with tf.device('/device:GPU:0'):
        inputs = Input((192, 192, 3), name='Input')
    
        conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', name='conv1_1')(inputs)
        conv1 = BatchNormalization(name='BN1_1')(conv1)
        conv1 = Activation('relu', name ='Activation1_1')(conv1)
        conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', name='conv1_2')(conv1)
        conv1 = BatchNormalization(name='BN1_2')(conv1)
        conv1 = Activation('relu', name ='Activation1_2')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='MaxPool1')(conv1)
    
        conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', name='conv2_1')(pool1)
        conv2 = BatchNormalization(name='BN2_1')(conv2)
        conv2 = Activation('relu', name ='Activation2_1')(conv2)
        conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', name='conv2_2')(conv2)
        conv2 = BatchNormalization(name='BN2_2')(conv2)
        conv2 = Activation('relu', name ='Activation2_2')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='MaxPool2')(conv2)
    
        conv3 = Conv2D(192, (3,3), padding='same', kernel_initializer='he_normal', name='conv3_1')(pool2)
        conv3 = BatchNormalization(name='BN3_1')(conv3)
        conv3 = Activation('relu', name ='Activation3_1')(conv3)
        conv3 = Conv2D(192, (3,3), padding='same', kernel_initializer='he_normal', name='conv3_2')(conv3)
        conv3 = BatchNormalization(name='BN3_2')(conv3)
        conv3 = Activation('relu', name ='Activation3_2')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='MaxPool3')(conv3)
    
        conv4 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', name='conv4_1')(pool3)
        conv4 = BatchNormalization(name='BN4_1')(conv4)
        conv4 = Activation('relu', name ='Activation4_1')(conv4)
        conv4 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', name='conv4_2')(conv4)
        conv4 = BatchNormalization(name='BN4_2')(conv4)
        conv4 = Activation('relu', name ='Activation4_2')(conv4)
        drop4 = Dropout(0.2, name='Dropout1')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='MaxPool4')(drop4)
    
        conv5 = Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', name='conv5_1')(pool4)
        conv5 = BatchNormalization(name='BN5_1')(conv5)
        conv5 = Activation('relu', name ='Activation5_1')(conv5)
        conv5 = Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', name='conv5_2')(conv5)
        conv5 = BatchNormalization(name='BN5_2')(conv5)
        conv5 = Activation('relu', name ='Activation5_2')(conv5)
        drop5 = Dropout(0.2, name='Dropout2')(conv5)
    
    with tf.device('/device:GPU:1'):
        up6 = Conv2D(384, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', name='conv6_1')(UpSampling2D(size = (2,2), name='Up1')(drop5))
        merge6 = keras.layers.Concatenate(name='concat1')([drop4, up6])
        conv6 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', name='conv6_2')(merge6)
        conv6 = BatchNormalization(name='BN6_1')(conv6)
        conv6 = Activation('relu', name ='Activation6_1')(conv6)
        conv6 = Conv2D(224, (3,3), padding='same', kernel_initializer='he_normal', name='conv6_3')(conv6)
        conv6 = BatchNormalization(name='BN6_2')(conv6)
        conv6 = Activation('relu', name ='Activation6_2')(conv6)
    
        up7 = Conv2D(224, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv7_1')(UpSampling2D(size = (2,2), name='Up2')(conv6))
        merge7 = keras.layers.Concatenate(name='concat2')([conv3, up7])
        conv7 = Conv2D(192, (3,3), padding='same', kernel_initializer='he_normal', name='conv7_2')(merge7)
        conv7 = BatchNormalization(name='BN7_1')(conv7)
        conv7 = Activation('relu', name ='Activation7_1')(conv7)
        conv7 = Conv2D(160, (3,3), padding='same', kernel_initializer='he_normal', name='conv7_3')(conv7)
        conv7 = BatchNormalization(name='BN7_2')(conv7)
        conv7 = Activation('relu', name ='Activation7_2')(conv7)
    
        up8 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv8_1')(UpSampling2D(size = (2,2), name='Up3')(conv7))
        merge8 = keras.layers.Concatenate(name='concat3')([conv2, up8])
        conv8 = Conv2D(96, (3,3), padding='same', kernel_initializer='he_normal', name='conv8_2')(merge8)
        conv8 = BatchNormalization(name='BN8_1')(conv8)
        conv8 = Activation('relu', name ='Activation8_1')(conv8)
        conv8 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', name='conv8_3')(conv8)
        conv8 = BatchNormalization(name='BN8_2')(conv8)
        conv8 = Activation('relu', name ='Activation8_2')(conv8)
    
        up9 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv9_1')(UpSampling2D(size = (2,2), name='Up4')(conv8))
        merge9 = keras.layers.Concatenate(name='concat4')([conv1, up9])
        conv9 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal', name='conv9_2')(merge9)
        conv9 = BatchNormalization(name='BN9_1')(conv9)
        conv9 = Activation('relu', name ='Activation9_1')(conv9)
        conv9 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal', name='conv9_3')(conv9)
        conv9 = BatchNormalization(name='BN9_2')(conv9)
        conv9 = Activation('relu', name ='Activation9_2')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid', name='conv10_1')(conv9)
    
        model = Model(inputs, conv10)

        model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.003, momentum=0.9, nesterov=True), metrics=[dice_coef,jaccard_coef,'acc'])

    return model

model = get_unet()

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint("-val_loss2_checkpoint.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6)
            ]

history = model.fit(trainData, trainMask, validation_data=(valData, valMask), batch_size=64, epochs=200, verbose=1, shuffle=True, callbacks=callbacks)

model.save('final2_attention_justconv.h5')

plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.savefig('lossAutoencoderjustconv2.png')

plt.figure(1)
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('Dice Coef Accuracy')
plt.xlabel('epoch')
plt.legend(['dice_coef', 'val_dice_coef'], loc='lower right')
plt.savefig('dice_coef_autoEncoderjustconv2.png')

plt.figure(2)
plt.plot(history.history['jaccard_coef'])
plt.plot(history.history['val_jaccard_coef'])
plt.title('Jaccard Coef Accuracy')
plt.xlabel('epoch')
plt.legend(['jaccard_coef', 'val_jaccard_coef'], loc='lower right')
plt.savefig('jaccard_coef_autoEncoderjustconv2.png')

plt.figure(3)
plt.plot(history.history['lr'])
plt.title('Learning Rate')
plt.xlabel('epoch')
plt.savefig('lr_autoencoderjustconv2.png')
