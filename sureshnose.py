#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 01:03:08 2018

@author: ck807
"""

import numpy as np

import tensorflow as tf

from keras.models import Model
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras.layers.convolutional import Conv2D, UpSampling2D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Input, Dropout, Dense, BatchNormalization, Flatten, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from residualblockwithlayernames import initial_conv_block1, Residual2, Residual3, Residual4, Residual5, Residual6
from residualblockwithlayernames import Residual7, Residual8, Residual9, Residual10, Residual11, Residual12
from residualblockwithlayernames import Residual13, Residual14, Residual15, Residual16, Residual17, Residual18, Residual19


from se import squeeze_excite_block

from layers import initial_conv_block, bottleneck_block_with_se

from resnet import _conv_bn_relu, _residual_block, basic_block, _bn_relu

import keras.backend as K

CHANNEL_AXIS = 3

trainData = np.load('train9DataRegressor2.npy')
trainLabel = np.load('train9LabelRegressor2.npy')
valData = np.load('val9DataRegressor2.npy')
valLabel = np.load('val9LabelRegressor2.npy')

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta
    
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    
    return tf.where(cond, squared_loss, linear_loss)

def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))

w = 10.0
e = 2.0
c = w - w * K.log(1 + (w/e))

def wingLoss(y_true, y_pred, w=w, e=e, c=c):
    error = y_true - y_pred
    cond = K.abs(error) < w
    true = w * (K.log(1 + (K.abs(error)/e)))
    otherwise = K.abs(error) - c
    return tf.where(cond, true, otherwise)


input = Input((192, 192, 3), name='Input')
           
conv1 = initial_conv_block1(input)
#conv1 = Residual2(32, 64, conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), name='MaxPool1')(conv1)
    
conv2 = Residual3(64, 128, pool1)
#conv2 = Residual4(128, 128, conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), name='MaxPool2')(conv2)
    
conv3 = Residual5(128, 256, pool2)
#conv3 = Residual6(256, 256, conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), name='MaxPool3')(conv3)
    
#conv4 = Residual7(256, 512, pool3)
#conv4 = Residual8(512, 512, conv4)
#drop4 = Dropout(0.2, name='Dropout1')(conv4)
#pool4 = MaxPooling2D(pool_size=(2, 2), name='MaxPool4')(drop4)

conv5 = Residual9(256, 512, pool3)
#conv5 = Residual10(512, 512, conv5)
drop5 = Dropout(0.2, name='Dropout2')(conv5)
    
#up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv1')(UpSampling2D(size = (2,2), name='Up1')(drop5))
#merge6 = keras.layers.Concatenate(name='Concat1')([drop4,up6])
#conv6 = Residual11(1024, 512, merge6)
#conv6 = Residual12(512, 512, conv6)
    
up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv2')(UpSampling2D(size = (2,2), name='Up2')(drop5))
merge7 = keras.layers.Concatenate(name='Concat2')([conv3,up7])
conv7 = Residual13(512, 256, merge7)
#conv7 = Residual14(256, 256, conv7)
    
up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv3')(UpSampling2D(size = (2,2), name='Up3')(conv7))
merge8 = keras.layers.Concatenate(name='Concat3')([conv2,up8])
conv8 = Residual15(256, 128, merge8)
#conv8 = Residual16(128, 128, conv8)
    
up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv4')(UpSampling2D(size = (2,2), name='Up4')(conv8))
merge9 = keras.layers.Concatenate(name='Concat4')([conv1,up9])
conv9 = Residual17(128, 64, merge9)
#conv10 = Residual18(64, 16, conv9)
#conv10 = Residual19(16, 1, conv10)
#conv11 = Conv2D(1, 1, activation = 'sigmoid', name='Output')(conv10)
    

conv1r = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(1, 1))(input)
  
block1 = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=True)(conv1r)
#se1 = squeeze_excite_block(block1)
gate1 = Activation('sigmoid')(conv9)
block1concat = keras.layers.Multiply()([block1, gate1])
#block1se = squeeze_excite_block(block1concat)
block1conv1 = Conv2D(64, (1,1), padding = 'same', kernel_initializer = 'he_normal')(block1concat)
block1conv1 = BatchNormalization(axis=CHANNEL_AXIS)(block1conv1)
block1conv1 = layers.LeakyReLU()(block1conv1)
block1conv2 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = 'he_normal')(block1conv1)
block1conv2 = BatchNormalization(axis=CHANNEL_AXIS)(block1conv2)
block1conv2 = layers.LeakyReLU()(block1conv2)
block1conv3 = Conv2D(64, (1,1), padding = 'same', kernel_initializer = 'he_normal')(block1conv2)
block1conv3 = BatchNormalization(axis=CHANNEL_AXIS)(block1conv3)
block1conv3 = layers.LeakyReLU()(block1conv3)
block1b = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=False)(block1conv3)
    
block2 = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=True)(block1b)
#se2 = squeeze_excite_block(block2)
gate2 = Activation('sigmoid')(conv8)
block2concat = keras.layers.Multiply()([block2, gate2])
#block2se = squeeze_excite_block(block2concat)
block2conv1 = Conv2D(128, (1,1), padding = 'same', kernel_initializer = 'he_normal')(block2concat)
block2conv1 = BatchNormalization(axis=CHANNEL_AXIS)(block2conv1)
block2conv1 = layers.LeakyReLU()(block2conv1)
block2conv2 = Conv2D(128, (3,3), padding = 'same', kernel_initializer = 'he_normal')(block2conv1)
block2conv2 = BatchNormalization(axis=CHANNEL_AXIS)(block2conv2)
block2conv2 = layers.LeakyReLU()(block2conv2)
block2conv3 = Conv2D(128, (1,1), padding = 'same', kernel_initializer = 'he_normal')(block2conv2)
block2conv3 = BatchNormalization(axis=CHANNEL_AXIS)(block2conv3)
block2conv3 = layers.LeakyReLU()(block2conv3)
block2b = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=False)(block2conv3)
  
block3 = _residual_block(basic_block, filters=256, repetitions=1, is_first_layer=True)(block2b)
#se3 = squeeze_excite_block(block3)
gate3 = Activation('sigmoid')(conv7)
block3concat = keras.layers.Multiply()([block3, gate3])
#block3se = squeeze_excite_block(block3concat)
block3conv1 = Conv2D(256, (1,1), padding = 'same', kernel_initializer = 'he_normal')(block3concat)
block3conv1 = BatchNormalization(axis=CHANNEL_AXIS)(block3conv1)
block3conv1 = layers.LeakyReLU()(block3conv1)
block3conv2 = Conv2D(256, (3,3), padding = 'same', kernel_initializer = 'he_normal')(block3conv1)
block3conv2 = BatchNormalization(axis=CHANNEL_AXIS)(block3conv2)
block3conv2 = layers.LeakyReLU()(block3conv2)
block3conv3 = Conv2D(256, (1,1), padding = 'same', kernel_initializer = 'he_normal')(block3conv2)
block3conv3 = BatchNormalization(axis=CHANNEL_AXIS)(block3conv2)
block3conv3 = layers.LeakyReLU()(block3conv2)
block3b = _residual_block(basic_block, filters=256, repetitions=1, is_first_layer=False)(block3conv3)

block4 = _residual_block(basic_block, filters=512, repetitions=1, is_first_layer=True)(block3b)
#block4se = squeeze_excite_block(block4)
#block4conv1 = Conv2D(512, (3,3), padding = 'same', kernel_initializer = 'he_normal')(block4se)
#block4conv1 = BatchNormalization(axis=CHANNEL_AXIS)(block4conv1)
#block4conv1 = layers.LeakyReLU()(block4conv1)
#block4conv2 = Conv2D(512, (3,3), padding = 'same', kernel_initializer = 'he_normal')(block4conv1)
#block4conv2 = BatchNormalization(axis=CHANNEL_AXIS)(block4conv2)
#block4conv2 = layers.LeakyReLU()(block4conv2)
block4b = _residual_block(basic_block, filters=512, repetitions=1, is_first_layer=False)(block4)
    
blockact = _bn_relu(block4b)
    
block_shape = K.int_shape(blockact)
poolr = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]), strides=(1, 1))(blockact)
    
flatten = Flatten()(poolr)
    
dense = Dense(units=18, kernel_initializer="he_normal", activation="linear")(flatten)

model = Model(inputs=input, outputs=dense)

#model.load_weights('resunet9_attention_checkpoint.h5', by_name=True)

model.summary()

model.compile(loss=wingLoss, optimizer='Adam', metrics=['mean_absolute_error'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=0.001),
    ModelCheckpoint("sureshresnetnose_val_loss_bottleneck_ldmk9_checkpoint.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, mode='auto', epsilon=0.001, cooldown=0, min_lr=0.5e-7)
            ]

history = model.fit(trainData, trainLabel, batch_size=4, epochs=200, validation_data=(valData, valLabel), shuffle=True, callbacks=callbacks)

model.save('sureshresnetnose_bottleneck_ldmk9_model.h5')

plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.savefig('sureshnose_ldmk_loss.png')

plt.figure(1)
plt.plot(history.history['lr'])
plt.title('Learning Rate')
plt.xlabel('epoch')
plt.savefig('sureshnose_ldmk_lr.png')

plt.figure(2)
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Mean Absolute Error Accuracy')
plt.xlabel('epoch')
plt.legend(['mean_absolute_error', 'val_mean_absolute_error'], loc='upper right')
plt.savefig('sureshnose_ldmk9_metric.png')
