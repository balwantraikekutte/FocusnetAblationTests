#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 23:29:27 2018

@author: ck807
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf 
import keras
from keras.models import Model
from keras.layers.core import Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D, SeparableConv2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from keras.layers import Input, Add, Concatenate
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K

from se import squeeze_excite_block

from layers import initial_conv_block, bottleneck_block, bottleneck_block_with_se
from layers import initial_SepConv_block, seperableConv_bottleneck_block_with_se

#-------------------------------------------------------------------------------------------------------------------------------------
print('Loading the data..')
trainData = np.load('trainData.npy')
trainMask = np.load('trainMask.npy')
valData = np.load('valData.npy')
valMask = np.load('valMask.npy')

print('PreProcessing the data..')
trainData = trainData.astype('float32')
trainDataMean = np.mean(trainData)
trainDataStd = np.std(trainData)

trainData -= trainDataMean
trainData /= trainDataStd

trainMask = trainMask.astype('float32')
trainMask /= 255.

valData = valData.astype('float32')

valData -= trainDataMean
valData /= trainDataStd

valMask = valMask.astype('float32')
valMask /= 255.
#-------------------------------------------------------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------------------------------------------------------
image_dim = 192
#-------------------------------------------------------------------------------------------------------------------------------------
input_image = Input(shape=(image_dim, image_dim, 3))
#-------------------------------------------------------------------------------------------------------------------------------------
with tf.device('/device:GPU:0'):
    init = initial_SepConv_block(input_image, weight_decay=5e-4) #192x192x16
    res1 = seperableConv_bottleneck_block_with_se(init, filters=32, cardinality=8, strides=1, weight_decay=5e-4)
    res1 = seperableConv_bottleneck_block_with_se(res1, filters=32, cardinality=32, strides=1, weight_decay=5e-4)  #192x192x32
    
    res2 = seperableConv_bottleneck_block_with_se(res1, filters=64, cardinality=8, strides=1, weight_decay=5e-4) 
    res2 = seperableConv_bottleneck_block_with_se(res2, filters=64, cardinality=32, strides=1, weight_decay=5e-4)  #192x192x64
    
    res3 = seperableConv_bottleneck_block_with_se(res2, filters=96, cardinality=8, strides=1, weight_decay=5e-4)  
    res3 = seperableConv_bottleneck_block_with_se(res3, filters=96, cardinality=32, strides=1, weight_decay=5e-4)  #192x192x96
    pool1 = MaxPooling2D(pool_size=(2,2))(res3)                                                                    #96x96x96
    
    res4 = seperableConv_bottleneck_block_with_se(pool1, filters=128, cardinality=8, strides=1, weight_decay=5e-4)
    res4 = seperableConv_bottleneck_block_with_se(res4, filters=128, cardinality=32, strides=1, weight_decay=5e-4) #96x96x128
    pool2 = MaxPooling2D(pool_size=(2,2))(res4)                                                                    #48x48x128         
    
    res5 = seperableConv_bottleneck_block_with_se(pool2, filters=160, cardinality=8, strides=1, weight_decay=5e-4)
    res5 = seperableConv_bottleneck_block_with_se(res5, filters=160, cardinality=32, strides=1, weight_decay=5e-4) #48x48x160
    
with tf.device('/device:GPU:1'):
    res6 = seperableConv_bottleneck_block_with_se(res5, filters=192, cardinality=8, strides=1, weight_decay=5e-4)
    res6 = seperableConv_bottleneck_block_with_se(res6, filters=192, cardinality=32, strides=1, weight_decay=5e-4) #48x48x192
    pool3 = MaxPooling2D(pool_size=(2,2))(res6)                                                                    #24x24x192
    
    res7 = seperableConv_bottleneck_block_with_se(pool3, filters=256, cardinality=8, strides=1, weight_decay=5e-4)
    res7 = seperableConv_bottleneck_block_with_se(res7, filters=256, cardinality=32, strides=1, weight_decay=5e-4)  #24x24x256
    pool4 = MaxPooling2D(pool_size=(2,2))(res7)                                                                     #12x12x256
    
    res8 = seperableConv_bottleneck_block_with_se(pool4, filters=384, cardinality=8, strides=1, weight_decay=5e-4)
    res8 = seperableConv_bottleneck_block_with_se(res8, filters=384, cardinality=32, strides=1, weight_decay=5e-4)  #12x12x384

with tf.device('/device:GPU:3'):
    up1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(res8)) #24x24x256
    merge1 = keras.layers.Add()([res7, up1])
    upres8 = seperableConv_bottleneck_block_with_se(merge1, filters=256, cardinality=32, strides=1, weight_decay=5e-4) #24x24x256
    
    up2 = Conv2D(192, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(upres8)) #48x48x192
    merge2 = keras.layers.Add()([res6, up2])
    upres7 = seperableConv_bottleneck_block_with_se(merge2, filters=192, cardinality=32, strides=1, weight_decay=5e-4) #48x48x192
    
    up3 = Conv2D(160, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upres7) #48x48x160
    merge3 = keras.layers.Add()([res5, up3])
    upres6 = seperableConv_bottleneck_block_with_se(merge3, filters=160, cardinality=32, strides=1, weight_decay=5e-4) #48x48x160

with tf.device('/device:GPU:3'):    
    up4 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(upres6)) # 96x96x128
    merge4 = keras.layers.Add()([res4, up4])
    upres5 = seperableConv_bottleneck_block_with_se(merge4, filters=128, cardinality=32, strides=1, weight_decay=5e-4) # 96x96x128
    
    up5 = Conv2D(96, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(upres5)) #192x192x96
    merge5 = keras.layers.Add()([res3, up5])
    upres4 = seperableConv_bottleneck_block_with_se(merge5, filters=96, cardinality=32, strides=1, weight_decay=5e-4) #192x192x96
    
    up6 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upres4) # 192x192x64
    merge6 = keras.layers.Add()([res2, up6])
    upres3 = seperableConv_bottleneck_block_with_se(merge6, filters=64, cardinality=32, strides=1, weight_decay=5e-4) #192x192x64
    
    up7 = Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upres3) #192x192x32
    merge7 = keras.layers.Add()([res1, up7])
    upres2 = seperableConv_bottleneck_block_with_se(merge7, filters=32, cardinality=32, strides=1, weight_decay=5e-4) #192x192x32
    upres1 = seperableConv_bottleneck_block_with_se(upres2, filters=32, cardinality=32, strides=1, weight_decay=5e-4) # 192x192x32
    outputConvAutoEncoder = Conv2D(1, (1,1), activation='sigmoid')(upres1) #192x192x1

model = Model(input_image, outputConvAutoEncoder)

model.summary()

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.003, momentum=0.9, nesterov=True), metrics=[dice_coef,jaccard_coef,'accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint("-val_loss_checkpoint.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6)
            ]

history = model.fit(trainData, trainMask, validation_data=(valData, valMask), batch_size=4, epochs=200, verbose=1, shuffle=True, callbacks=callbacks)

model.save('final_Attention.h5')

#plt.plot(history.history['lr'])
#plt.title('Learning Rate')
#plt.xlabel('epoch')
#plt.show()
#
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('accuracy')
#plt.xlabel('epoch')
#plt.legend(['acc', 'val_acc'], loc='upper left')
#plt.show()
#
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('loss')
#plt.xlabel('epoch')
#plt.legend(['loss', 'val_loss'], loc='upper right')
#plt.show()

