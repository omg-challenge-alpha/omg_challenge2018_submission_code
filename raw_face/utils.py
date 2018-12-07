
import numpy as np

import tensorflow as tf
import keras.backend as K
import keras.callbacks as cb


PATIENCE=3


import time

day_time = time.strftime("%Y-%m-%d_%H_%M_%S")

def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    true_variance = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_variance = np.var(y_pred)

    rho,_ = pearsonr(y_pred,y_true)
    std_predictions = np.std(y_pred)
    std_gt = np.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (
       std_predictions ** 2 + std_gt ** 2 +
       (pred_mean - true_mean) ** 2)

    return ccc, rho
  
  
  
def ccc_error(y_true, y_pred):
    true_mean = K.mean(y_true)
    true_variance = K.var(y_true)
    pred_mean = K.mean(y_pred)
    pred_variance = K.var(y_pred)

    x = y_true - true_mean
    y = y_pred - pred_mean
    rho = K.sum(x * y) / K.sqrt(K.sum(x**2) * K.sum(y**2))
    
    std_predictions = K.std(y_pred)
    std_gt = K.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (
       std_predictions ** 2 + std_gt ** 2 +
       (pred_mean - true_mean) ** 2)
    return 1-ccc


def norm_pred(lbl,pred):

	s0 = np.std(lbl.flatten())
	V = pred.flatten()
	m1 = np.mean(pred.flatten())
	s1 = np.std(pred.flatten())
	m0 = np.mean(lbl.flatten())

	norm_pred = s0*(V-m1)/s1+m0

	return norm_pred


def make_id_vector(str_n_s,sbj_n_s,lbl_path):
    
    id_s = []

    for str_n in str_n_s:

        for sbj_n in sbj_n_s:

            print(sbj_n,str_n)

            lbl = np.loadtxt(lbl_path.format(sbj_n,str_n),skiprows=1)

            id = np.zeros([len(lbl),len(sbj_n_s)])
            id[:,sbj_n-1]=1

            id_s.append(id)
            
    return id_s



class Metrics(cb.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(model.predict(X_val))
    
        ccc_result, rho_result =  ccc(y_val, y_predict)
        
        self._data.append({
           'ccc': ccc_result,
           'rho': rho_result
        })
        print("ccc = %f,  pearson=%f" % (ccc_result[0], rho_result[0]) )
        return

    def get_data(self):
        return self._data

metrics = Metrics()




class light_generator():
  
  def __init__(self,x,y,seq_len,batch_size):
    
    self.x = x
    self.y = y
    
    self.seq_len = seq_len
    self.sample_size = self.x.shape[0]
    
    self.h = self.x.shape[1]
    self.w = self.x.shape[2]
    self.c = self.x.shape[3]
    
    self.idx_s = np.arange(self.sample_size-self.seq_len)
    self.batch_size = batch_size
    self.stp_per_epoch = int(self.sample_size/self.batch_size)
    
   
  def generate(self):
    
    while True:
      
      for b in range(self.stp_per_epoch):

        np.random.shuffle(self.idx_s)
        rnd_idx = self.idx_s[:self.batch_size]

        
        xb = np.empty([self.batch_size,self.seq_len,self.h,self.w,self.c])
        yb = np.empty([self.batch_size,1])
        
        for i in range(len(rnd_idx)):
          
          ri = rnd_idx[i]
          xb[i,:,:,:,:] = self.x[ri:ri+self.seq_len,:,:,:]
          yb[i,:] = self.y[ri+self.seq_len,:]

        yield xb, yb




class light_id_generator():
  
  def __init__(self,x,y,z,seq_len,batch_size):
    
    self.x = x
    self.y = y
    self.z = z
    
    self.seq_len = seq_len
    self.sample_size = self.x.shape[0]
    
    self.h = self.x.shape[1]
    self.w = self.x.shape[2]
    self.c = self.x.shape[3]
    
    self.idx_s = np.arange(self.sample_size-self.seq_len)
    self.batch_size = batch_size
    self.stp_per_epoch = int(self.sample_size/self.batch_size)
    
   
  def generate(self):
    
    while True:
      
      for b in range(self.stp_per_epoch):

        np.random.shuffle(self.idx_s)
        rnd_idx = self.idx_s[:self.batch_size]

        
        xb = np.empty([self.batch_size,self.seq_len,self.h,self.w,self.c])
        yb = np.empty([self.batch_size,1])
        zb = np.empty([self.batch_size,self.z.shape[1]])
        
        for i in range(len(rnd_idx)):
          
          ri = rnd_idx[i]
          xb[i,:,:,:,:] = self.x[ri:ri+self.seq_len,:,:,:]
          yb[i,:] = self.y[ri+self.seq_len,:]
          zb[i,:] = self.z[ri+self.seq_len,:]

        yield [xb,zb], yb