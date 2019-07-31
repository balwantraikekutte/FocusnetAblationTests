#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:58:40 2018

@author: ck807
"""

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.models import Model
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from layers import initial_conv_block, bottleneck_block_with_se

print('Loading the dataset..')
trainData = np.load('trainDataRegressor.npy')
trainLabel = np.load('trainLabelRegressor.npy')
valData = np.load('valDataRegressor.npy')
valLabel = np.load('valLabelRegressor.npy')

noutput = 20
image_dim = 192

print('Building the model..')
with tf.device('/device:GPU:2'):
    input_image = Input(shape=(image_dim, image_dim, 3))

    x = initial_conv_block(input_image, weight_decay=5e-4)   #192x192x64

    x = bottleneck_block_with_se(x, filters=96, cardinality=32, strides=1, weight_decay=5e-4)    #192x192x192
    x = bottleneck_block_with_se(x, filters=96, cardinality=32, strides=1, weight_decay=5e-4)
    x = bottleneck_block_with_se(x, filters=96, cardinality=32, strides=1, weight_decay=5e-4)    #192x192x192

    x = bottleneck_block_with_se(x, filters=128, cardinality=32, strides=2, weight_decay=5e-4)   #96x96x256
    x = bottleneck_block_with_se(x, filters=128, cardinality=32, strides=1, weight_decay=5e-4)
    x = bottleneck_block_with_se(x, filters=128, cardinality=32, strides=1, weight_decay=5e-4)   #96x96x256

    x = bottleneck_block_with_se(x, filters=192, cardinality=32, strides=2, weight_decay=5e-4)   #48x48x384
    x = bottleneck_block_with_se(x, filters=192, cardinality=32, strides=1, weight_decay=5e-4)
    x = bottleneck_block_with_se(x, filters=192, cardinality=32, strides=1, weight_decay=5e-4)   #48x48x384

    x = bottleneck_block_with_se(x, filters=256, cardinality=32, strides=2, weight_decay=5e-4)   #24x24x512
    x = bottleneck_block_with_se(x, filters=256, cardinality=32, strides=1, weight_decay=5e-4)
    x = bottleneck_block_with_se(x, filters=256, cardinality=32, strides=1, weight_decay=5e-4)   #24x24x512

with tf.device('/device:GPU:3'):
    x = bottleneck_block_with_se(x, filters=384, cardinality=32, strides=2, weight_decay=5e-4)   #12x12x768
    x = bottleneck_block_with_se(x, filters=384, cardinality=32, strides=1, weight_decay=5e-4)
    x = bottleneck_block_with_se(x, filters=384, cardinality=32, strides=1, weight_decay=5e-4)   #12x12x768

    x = bottleneck_block_with_se(x, filters=512, cardinality=32, strides=2, weight_decay=5e-4)   #6x6x1024
    x = bottleneck_block_with_se(x, filters=512, cardinality=32, strides=1, weight_decay=5e-4)
    x = bottleneck_block_with_se(x, filters=512, cardinality=32, strides=1, weight_decay=5e-4)   #6x6x1024

    x = GlobalAveragePooling2D()(x)

    output = Dense(noutput, use_bias=False, kernel_regularizer=l2(5e-4), kernel_initializer='he_normal', activation='softmax')(x)

    model = Model(input_image, output)
    
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.003, momentum=0.9, nesterov=True), metrics=['mae'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint("-val_loss_checkpoint-regressorwithoutattention.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6)
            ]

history = model.fit(trainData, trainLabel, validation_data=(valData, valLabel), batch_size=4, epochs=200, verbose=1, shuffle=True, callbacks=callbacks)

model.save('regressorwithoutattention.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.savefig('lossregressorwithoutattention.png')

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Mean Absolute Error Accuracy')
plt.xlabel('epoch')
plt.legend(['mae', 'val_mae'], loc='upper right')
plt.savefig('mae_regressorwithoutattention.png')

plt.plot(history.history['lr'])
plt.title('Learning Rate')
plt.xlabel('epoch')
plt.savefig('lr_regressorwithoutattention.png')
