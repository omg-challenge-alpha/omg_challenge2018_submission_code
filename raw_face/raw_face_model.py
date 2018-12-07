
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, Dense, Dropout, Flatten, LSTM, Reshape, TimeDistributed, Input, concatenate, BatchNormalization
from keras.models import Sequential, load_model, Model
from keras import losses
from keras import optimizers
from keras.utils import to_categorical, Sequence




class conv_3d_id_model():
  
  
  
  def __init__(self,seq_len,img_x,img_y,ch_n,id_len):
    
    self.seq_len = seq_len
    self.img_x = img_x
    self.img_y = img_y
    self.ch_n = ch_n
    self.id_len = id_len
    
  
  
  def create(self):
    
  
    main_input = Input(shape=(self.seq_len,self.img_x,self.img_y,self.ch_n), name='main_input')

    layer = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(main_input)
    layer = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer) 

    layer = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(layer)
    layer = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer) 

    layer = Flatten()(layer)
    layer = Dense(512,activation='relu', name='conv_out')(layer)


    aux_input = Input(shape=(self.id_len,), name='aux_input')
    
    aux_layer = Dense(3,activation='relu')(aux_input)

    layer = concatenate([layer,aux_layer])
    layer = Dense(128,activation='relu')(layer)
    layer = Dropout(0.6)(layer)
    layer = Dense(32,activation='relu')(layer)
    layer = Dropout(0.6)(layer)
    reg_out = Dense(1,activation='linear',name='reg_out')(layer)


    reg_conv_3d_model_double_in = Model(inputs=[main_input, aux_input], outputs=[reg_out])
    
    return reg_conv_3d_model_double_in


