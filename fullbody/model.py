from resnet3d import Resnet3DBuilder

from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, Dense, Dropout, Flatten, LSTM, Reshape, TimeDistributed, InputLayer
from keras.models import Sequential, load_model, Model

import numpy as np
import tensorflow as tf
import keras.backend as K
import keras.callbacks as cb



def create_reg_resnet18_3D(img_x,img_y,ch_n,seq_len,tgt_size):

    resnet18_3D = Resnet3DBuilder.build_resnet_18((seq_len, img_x, img_y, ch_n), tgt_size)
    resnet18_3D.layers.pop()
    layer = Dense(32,activation='relu')(resnet18_3D.output)
    resnet18_3D.layers[-1].outbound_nodes = []
    resnet18_3D.outputs = [resnet18_3D.layers[-1].output]
    output = resnet18_3D.get_layer('flatten_1').output
    output = Dense(32, activation='relu')(output) 
    output = Dense(1, activation='linear')(output)
    reg_resnet18_3D = Model(resnet18_3D.input, output)
    
    return reg_resnet18_3D