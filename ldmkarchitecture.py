#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:49:39 2018

@author: ck807
"""

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
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K

from se import squeeze_excite_block

from layers import initial_conv_block, bottleneck_block, bottleneck_block_with_se
from layers import initial_SepConv_block, seperableConv_bottleneck_block_with_se


#-------------------------------------------------------------------------------------------------------------------------------------
noutput = 28
image_dim = 192
#-------------------------------------------------------------------------------------------------------------------------------------
input_image = Input(shape=(image_dim, image_dim, 3))
#-------------------------------------------------------------------------------------------------------------------------------------
init = initial_SepConv_block(input_image, weight_decay=5e-4) #196x196x16
res1 = seperableConv_bottleneck_block_with_se(init, filters=32, cardinality=8, strides=1, weight_decay=5e-4)
res1 = seperableConv_bottleneck_block_with_se(res1, filters=32, cardinality=32, strides=1, weight_decay=5e-4)  #196x196x32

res2 = seperableConv_bottleneck_block_with_se(res1, filters=64, cardinality=8, strides=1, weight_decay=5e-4) 
res2 = seperableConv_bottleneck_block_with_se(res2, filters=64, cardinality=32, strides=1, weight_decay=5e-4)  #196x196x64

res3 = seperableConv_bottleneck_block_with_se(res2, filters=96, cardinality=8, strides=1, weight_decay=5e-4)  
res3 = seperableConv_bottleneck_block_with_se(res3, filters=96, cardinality=32, strides=1, weight_decay=5e-4) 
res3 = MaxPooling2D(pool_size=(2,2))(res3)                                                                     #98x98x96

res4 = seperableConv_bottleneck_block_with_se(res3, filters=128, cardinality=8, strides=1, weight_decay=5e-4)
res4 = seperableConv_bottleneck_block_with_se(res4, filters=128, cardinality=32, strides=1, weight_decay=5e-4)
res4 = MaxPooling2D(pool_size=(2,2))(res4)                                                                     #49x49x128         

res5 = seperableConv_bottleneck_block_with_se(res4, filters=160, cardinality=8, strides=1, weight_decay=5e-4)
res5 = seperableConv_bottleneck_block_with_se(res5, filters=160, cardinality=32, strides=1, weight_decay=5e-4) #49x49x160

res6 = seperableConv_bottleneck_block_with_se(res5, filters=192, cardinality=8, strides=1, weight_decay=5e-4)
res6 = seperableConv_bottleneck_block_with_se(res6, filters=192, cardinality=32, strides=1, weight_decay=5e-4)
res6 = MaxPooling2D(pool_size=(2,2))(res6)                                                                     #24x24x192

res7 = seperableConv_bottleneck_block_with_se(res6, filters=256, cardinality=8, strides=1, weight_decay=5e-4)
res7 = seperableConv_bottleneck_block_with_se(res7, filters=256, cardinality=32, strides=1, weight_decay=5e-4)
res7 = MaxPooling2D(pool_size=(2,2))(res7)                                                                     #12x12x256

res8 = seperableConv_bottleneck_block_with_se(res7, filters=384, cardinality=8, strides=1, weight_decay=5e-4)
res8 = seperableConv_bottleneck_block_with_se(res8, filters=384, cardinality=32, strides=1, weight_decay=5e-4)
res8 = MaxPooling2D(pool_size=(2,2))(res8)                                                                     #6x6x384

res9 = seperableConv_bottleneck_block_with_se(res8, filters=512, cardinality=8, strides=1, weight_decay=5e-4)
res9 = seperableConv_bottleneck_block_with_se(res9, filters=512, cardinality=32, strides=1, weight_decay=5e-4)

up1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(res9))
merge1 = keras.layers.Add()([res7, up1])
upres9 = seperableConv_bottleneck_block_with_se(merge1, filters=256, cardinality=32, strides=1, weight_decay=5e-4)

up2 = Conv2D(192, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(upres9))
merge2 = keras.layers.Add()([res6, up2])
upres8 = seperableConv_bottleneck_block_with_se(merge2, filters=196, cardinality=32, strides=1, weight_decay=5e-4)

up3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(upres8))
merge3 = keras.layers.Add()([res7, up3])
upres7 = seperableConv_bottleneck_block_with_se(merge3, filters=256, cardinality=32, strides=1, weight_decay=5e-4)

up4 = Conv2D(192, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(upres7))
merge4 = keras.layers.Add()([res6, up4])
upres6 = seperableConv_bottleneck_block_with_se(merge4, filters=192, cardinality=32, strides=1, weight_decay=5e-4)

up5 = Conv2D(160, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(upres6))
merge5 = keras.layers.Add()([res5, up5])
upres5 = seperableConv_bottleneck_block_with_se(merge5, filters=160, cardinality=32, strides=1, weight_decay=5e-4)

up6 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(upres5))
merge6 = keras.layers.Add()([res4, up6])
upres4 = seperableConv_bottleneck_block_with_se(merge6, filters=96, cardinality=32, strides=1, weight_decay=5e-4)

up7 = Conv2D(96, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(upres4))
merge7 = keras.layers.Add()([res3, up7])
upres3 = seperableConv_bottleneck_block_with_se(merge7, filters=64, cardinality=32, strides=1, weight_decay=5e-4)

up7 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(upres3))
merge7 = keras.layers.Add()([res2, up7])
upres2 = seperableConv_bottleneck_block_with_se(merge7, filters=32, cardinality=32, strides=1, weight_decay=5e-4)

up8 = Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(upres2))
merge8 = keras.layers.Add()([res1, up8])
upres1 = seperableConv_bottleneck_block_with_se(merge8, filters=32, cardinality=32, strides=1, weight_decay=5e-4)
upres1 = seperableConv_bottleneck_block_with_se(upres1, filters=1, cardinality=1, strides=1, weight_decay=5e-4)
outputConvAutoEncoder = Conv2D(1, (1,1), activation='sigmoid')(upres1)
#-------------------------------------------------------------------------------------------------------------------------------------
x = initial_conv_block(input_image, weight_decay=5e-4)

x = bottleneck_block_with_se(x, filters=96, cardinality=32, strides=1, weight_decay=5e-4)    # 196x196
x = bottleneck_block_with_se(x, filters=96, cardinality=32, strides=1, weight_decay=5e-4)
x = bottleneck_block_with_se(x, filters=96, cardinality=32, strides=1, weight_decay=5e-4)
x = MaxPooling2D(pool_size=(2,2))(x)

x = bottleneck_block_with_se(x, filters=128, cardinality=32, strides=1, weight_decay=5e-4)  # 49x49
x = bottleneck_block_with_se(x, filters=128, cardinality=32, strides=1, weight_decay=5e-4)
x = bottleneck_block_with_se(x, filters=128, cardinality=32, strides=1, weight_decay=5e-4)
x = MaxPooling2D(pool_size=(2,2))(x)

x = bottleneck_block_with_se(x, filters=192, cardinality=32, strides=1, weight_decay=5e-4)  # 25x25
x = bottleneck_block_with_se(x, filters=192, cardinality=32, strides=1, weight_decay=5e-4)
x = bottleneck_block_with_se(x, filters=192, cardinality=32, strides=1, weight_decay=5e-4)
x = MaxPooling2D(pool_size=(2,2))(x)

x = bottleneck_block_with_se(x, filters=256, cardinality=32, strides=1, weight_decay=5e-4)  # 13x13
x = bottleneck_block_with_se(x, filters=256, cardinality=32, strides=1, weight_decay=5e-4)
x = bottleneck_block_with_se(x, filters=256, cardinality=32, strides=1, weight_decay=5e-4)
x = MaxPooling2D(pool_size=(2,2))(x)

x = bottleneck_block_with_se(x, filters=384, cardinality=32, strides=1, weight_decay=5e-4)  # 7x7
x = bottleneck_block_with_se(x, filters=384, cardinality=32, strides=1, weight_decay=5e-4)
x = bottleneck_block_with_se(x, filters=384, cardinality=32, strides=1, weight_decay=5e-4)
x = MaxPooling2D(pool_size=(2,2))(x)

x = bottleneck_block_with_se(x, filters=512, cardinality=32, strides=1, weight_decay=5e-4)  # 4x4
x = bottleneck_block_with_se(x, filters=512, cardinality=32, strides=1, weight_decay=5e-4)
x = bottleneck_block_with_se(x, filters=512, cardinality=32, strides=1, weight_decay=5e-4)

x = GlobalAveragePooling2D()(x)

output = Dense(noutput, use_bias=False, kernel_regularizer=l2(5e-4), kernel_initializer='he_normal', activation='softmax')(x)

model = Model(input_image, output)
#-------------------------------------------------------------------------------------------------------------------------------------