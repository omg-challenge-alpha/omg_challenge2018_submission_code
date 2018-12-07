from model import *
from utils import *

from keras import losses
from keras import optimizers

import numpy as np
import tensorflow as tf
import keras.backend as K
import keras.callbacks as cb



### definition of constants for network and image preprocessing

seq_len = 16
img_x = 128
img_y = 128
ch_n = 1
tgt_size = 1
down_sampling = 5

img_path = '/OMG_Empathy2019/full_body/full_body/Subject_{0}_Story_{1}/Subject_img/'
lbl_path = '/OMG_Empathy2019/labels/Subject_{0}_Story_{1}.csv'
save_path = '/OMG_empathy_challange/models/trained/global/'




### building the model

K.clear_session()
m = create_reg_resnet18_3D(img_x,img_y,ch_n,seq_len,tgt_size)
opti = optimizers.Adam(lr=0.0001)
m.compile(loss=ccc_error, optimizer=opti)

print('3d resent 16 model loaded')


### loading dataset and preprocessing

##### load training and make generator

sbj_n_s = range(1,11)
str_n_s = [1,4,5,8]

lbl_tr = np.concatenate([np.loadtxt(lbl_path.format(sbj_n,str_n),skiprows=1)[::down_sampling] for str_n in str_n_s for sbj_n in sbj_n_s]).reshape(-1,1)
img_tr = create_img_dataset(lbl_tr.shape[0],img_x ,img_y, ch_n ,str_n_s,sbj_n_s,down_sampling)

print('train images loaded with shape: ',img_tr.shape)
print('train labels loaded with shape: ',lbl_tr.shape)

lw_gen_tr = light_generator(img_tr[:],lbl_tr[:],seq_len,batch_size)
steps_per_epoch_tr = lw_gen_tr.stp_per_epoch



##### load validation and make generator

sbj_n_s = range(1,11)
str_n_s = [2]

lbl_vl = np.concatenate([np.loadtxt(lbl_path.format(sbj_n,str_n),skiprows=1)[::down_sampling] for str_n in str_n_s for sbj_n in sbj_n_s]).reshape(-1,1)
img_vl = create_img_dataset(lbl_vl.shape[0],img_x ,img_y, ch_n,str_n_s,sbj_n_s,down_sampling)

print('valid images loaded with shape: ',img_tr.shape)
print('valid labels loaded with shape: ',lbl_tr.shape)

lw_gen_vl = light_generator(img_vl,lbl_vl,seq_len,batch_size)
steps_per_epoch_vl = lw_gen_vl.stp_per_epoch



### training

batch_size = 128
epochs = 200


save_name = save_path+'resnet_ccc/resnet_3D_global_regression.{epoch:02d}-{val_loss:.2f}.hdf5'



bckup_callback = cb.ModelCheckpoint(save_name, 
                                    monitor='val_loss', 
                                    verbose=0, 
                                    save_best_only=True, 
                                    save_weights_only=False, 
                                    mode='auto', 
                                    period=1)



stop_callback = cb.EarlyStopping(monitor='val_loss', patience=10)


callbacks_list = [
    
    bckup_callback,
    stop_callback,
    cb.TensorBoard(log_dir="logs/" + day_time)
]



m.fit_generator(   lw_gen_tr.generate(),
                   steps_per_epoch=steps_per_epoch_tr, 
                   epochs=epochs,
                   callbacks=callbacks_list,
                   validation_data = lw_gen_vl.generate(),
                   validation_steps = steps_per_epoch_vl,
                   verbose=True,
                   shuffle=True) ### add callback to CCC here






