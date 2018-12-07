import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from scipy.signal import butter, lfilter, freqz
from scipy.signal import argrelextrema

#try with minmax
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM, Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Reshape, BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
import keras.backend as K
from keras.callbacks import CSVLogger

from contextlib import redirect_stdout
from scipy.optimize import minimize

from scipy.signal import savgol_filter
from models.attlayer import AttentionWeightedAverage
import time
#from time import time



import re
def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def Y_preprocessing(Y_video):
    Y_video -= np.mean(Y_video, axis=0)
    Y_video /= np.std(Y_video, axis=0)
    return Y_video

def X_preprocessing(X_video):
    X_video -= np.mean(X_video, axis=0)
    X_video /= np.std(X_video, axis=0)
    return X_video

def Y_window_labels(Y, window_size):
    return Y

def X_window_samples(X, window_size):
    N_features = X.shape[1]
    X_out = np.zeros([len(X), window_size, N_features])

    window_index = window_size -1
    for i in range(window_index):
        X_out[i] = X[window_index,:]

    for i in range(window_index, len(X_out)):
        X_out[i] = X[i-window_index:i+1,:]
    return X_out

def create_Y(subjects_list, stories_list):
    total_n_videos   = len(subjects_list)*len(stories_list)

    # storing indexes
    indexes = np.zeros(total_n_videos+1,dtype=int)
    indexes[0] = 0

    i = 0
    for subject in subjects_list:
        for story in stories_list:
            video_name = "Subject_" + str(subject) + "_Story_" + str(story)

            Y_video = np.loadtxt(base_path_Y + video_name + ".csv", delimiter=",", skiprows=1)[:,np.newaxis]

            #preprocessing
            Y_video = Y_preprocessing(Y_video)
            #window sampling
            Y_video = Y_window_labels(Y_video, window_size)

            indexes[i+1] = len(Y_video) + indexes[i]

            if i==0:
                Y = Y_video
            else:
                Y = np.concatenate([Y, Y_video])

            i +=1
    return Y, indexes

def create_X(subjects_list, stories_list, indexes):

    total_n_videos   = len(subjects_list)*len(stories_list)

    i = 0
    for subject in subjects_list:
        for story in stories_list:
            video_name = "Subject_" + str(subject) + "_Story_" + str(story)

            print("\t- Processing landmarks from video: " + video_name + ", " + str(i+1) + "/" + str(total_n_videos))

            if subject_data:
                X_video = np.loadtxt(base_path_X + video_name + ".mp4/Subject_face_landmarks/landmarksSubject.csv",
                                 delimiter=",")
            elif actor_data:
                X_video = np.loadtxt(base_path_X + video_name + ".mp4/Actor_face_landmarks/landmarksActor.csv",
                                 delimiter=",")

            # preprocessing
            X_video = X_preprocessing(X_video)
            #window sampling
            X_video = X_window_samples(X_video, window_size)

            print("\t", X_video.shape)

            if i==0:
                X = np.zeros([indexes[-1], window_size, X_video.shape[-1]])

            X[indexes[i]:indexes[i+1]] = X_video
            del X_video

            i +=1

    return X


def load_dataset(window_size):

    ## Y TRAINING
    print("Processing Y_training")
    Y_training, indexes_training = create_Y(subjects_training, stories_training)
    print("Y_training shape = ", Y_training.shape)
    print("indexes_training = ", indexes_training)

    ## Y VALIDATION
    print("\nProcessing Y_validation")
    Y_validation, indexes_validation = create_Y(subjects_validation, stories_validation)
    print("Y_validation shape = ", Y_validation.shape)
    print("indexes_validation = ", indexes_validation)

    ## X TRAINING
    print("\nProcessing X_training")
    X_training = create_X(subjects_training, stories_training, indexes_training)
    print("X_training shape = ", X_training.shape)

    ## X VALIDATION
    print("\nProcessing X_validation")
    X_validation = create_X(subjects_validation, stories_validation, indexes_validation)
    print("X_validation shape = ", X_validation.shape)

    return X_training, Y_training, X_validation, Y_validation, indexes_training, indexes_validation







def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    #true_variance = np.var(y_true)
    pred_mean = np.mean(y_pred)
    #pred_variance = np.var(y_pred)

    rho,_ = pearsonr(y_pred,y_true)
    std_predictions = np.std(y_pred)
    std_gt = np.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (
       std_predictions ** 2 + std_gt ** 2 +
       (pred_mean - true_mean) ** 2)

    return ccc, rho



# losses
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def ccc_error(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

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

def pearson_error(y_true, y_pred):
    true_mean = K.mean(y_true)
    true_variance = K.var(y_true)
    pred_mean = K.mean(y_pred)
    pred_variance = K.var(y_pred)

    x = y_true - true_mean
    y = y_pred - pred_mean
    rho = K.sum(x * y) / K.sqrt(K.sum(x**2) * K.sum(y**2))
    return 1-rho



def f_trick(Y_train, preds):
    Y_train_flat = Y_train.flatten()
    preds_flat = preds.flatten()
    s0 = np.std(Y_train_flat)
    V = preds_flat
    m1 = np.mean(preds_flat)
    s1 = np.std(preds_flat)
    m0 = np.mean(Y_train_flat)
    norm_preds = s0*(V-m1)/s1+m0
    return norm_preds


def CCC(x,y):
    sx = np.std(x)
    sy = np.std(y)
    mx = np.mean(x)
    my = np.mean(y)
    rho = pearsonr(x,y)[0]
    return 2*rho*sx*sy/(sx**2+sy**2+(mx-my)**2)

def ComputeAll(datas,mus,sigmas, cutoff,order,fs=25):
    cccs = []
    for i,d in enumerate(datas):
        y = butter_lowpass_filter_bidirectional(d[:,1],cutoff,fs,order)
        my = np.mean(y)
        sy = np.std(y)
        y = (y-my)*sigmas[i]/sy+mus[i]
        cccs.append(CCC(d[:,0],y))
    return(1.0-np.mean(cccs))

lambdaM = lambda datas,mus,sigmas,order:(lambda cutoff:ComputeAll(datas,mus,sigmas,cutoff,order))

def OptimMCutoffA(datas, mus, sigmas, order=5,start=.01):
    ff = lambdaM(datas, mus, sigmas, order)
    cutoff = minimize(ff, np.array([start]),
                     method='L-BFGS-B',
                     options={'disp': True},
                     bounds=[(1e-8,10)]).x
    value = ff(cutoff)
    return((cutoff[0],1.0-value))

def OptimMCutoffB(datas, mus, sigmas, order=5,ngrid=5):
    cutoffs = np.exp(np.linspace(start=np.log(.0001), stop=np.log(10), num=ngrid))
    best = (0,2e10)
    for i in range(ngrid):
        cutoff,value = OptimMCutoffA(datas,mus,sigmas, order,start=cutoffs[i])
        if 1-value <= best[1]:
            best = (cutoff,value)
    return(best)


def OptimMOrder(datas,mus,sigmas,orders=range(1,7)):
    cutoffs,rhos = zip(*[OptimMCutoffB(datas,mus,sigmas, o) for o in orders])
    i=np.argmax(np.array(rhos))
    return((orders[i],cutoffs[i],rhos[i]))


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []
        self.best_ccc = -1

    def on_epoch_end(self, batch, logs={}):
        #          X_val, y_val = self.validation_data[:len(modalities)], self.validation_data[len(modalities)]
        X_val, y_val = self.validation_data[0], self.validation_data[1]

        y_predict = np.asarray(model.predict(X_val, batch_size=batch_size))

        ccc_result, rho_result =  ccc(y_val, y_predict)

        self._data.append({
           'ccc': ccc_result,
           'rho': rho_result
        })

        if ccc_result > self.best_ccc:
            model.save(filepath)
            self.best_ccc = ccc_result
            print("ccc = %f,  pearson=%f,  (new model!)" % (ccc_result[0], rho_result[0]) )
        else:
            print("ccc = %f,  pearson=%f" % (ccc_result[0], rho_result[0]) )

        return

    def get_data(self):
        return self._data



def Derivative(x):
    x1 = x[1:-1]-x[:-2]
    x2 = x[2:]-x[1:-1]
    return((x1+x2)/2)

def Shoulder(x):
    plt.figure()
    power,freq=plt.psd(x,Fs=25)
    plt.show()
    mm=argrelextrema(Derivative(power), np.greater)
    return(freq[mm[0][0]+1])

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter_bidirectional(data, cutoff=0.1, fs=25, order=1):
    y_first_pass = butter_lowpass_filter(data[::-1].flatten(), cutoff, fs, order)
    y_second_pass = butter_lowpass_filter(y_first_pass[::-1].flatten(), cutoff, fs, order)
    return y_second_pass

def entropy_norm_dist(data):
    std = np.std(data)
    return 0.5 * np.log(2*np.pi*np.exp(1)*std*std)

def best_filter_values(Y_train, preds_train, ccc_tmp_tag=False):
    best_cutoff = 1
    best_order = 1
    best_ccc = 0

    shoulder_freq = Shoulder(preds_train)


    min_grid = np.log(0.001)
    max_grid = np.log(.1)
    nb_samples = 500
    r = np.random.rand(nb_samples)*(max_grid-min_grid) + min_grid
    exp_r = np.exp(r)
    exp_r = np.sort(exp_r)


    cutoff_grid = exp_r #np.logspace(-4, -2, 1000)#(0.0, 0.05, 0.0001)
    print(cutoff_grid.shape)
    order_grid = [1,2,3,4,5,6,7,8]
    freq_ccc_pearson_H_list =np.zeros((len(cutoff_grid)*len(order_grid),7))

    i = 0
    for order in order_grid:
        print("order " + str(order))
        for cutoff in cutoff_grid:
            if cutoff > 0:
                filtered_preds = butter_lowpass_filter_bidirectional(preds_train, cutoff=cutoff, order=order)
                ccc_tmp = ccc(Y_train, filtered_preds.flatten())

                if ccc_tmp_tag==True:
                    ccc_pearson = [ccc_tmp[0], ccc_tmp[1]]
                    H_raw       = entropy_norm_dist(preds_train.flatten())
                    H_filtered  = entropy_norm_dist(filtered_preds.flatten())

                    out = [order] + [cutoff] + ccc_pearson + [H_raw] + [H_filtered] + [shoulder_freq]
                    freq_ccc_pearson_H_list[i,:] = out
                    i+=1

                if ccc_tmp[1] > best_ccc:
                    #print(order,cutoff)
                    #print(ccc_tmp,best_ccc)
                    best_cutoff = cutoff
                    best_order = order
                    best_ccc = ccc_tmp[1]

    if ccc_tmp_tag==True:
        return np.array(freq_ccc_pearson_H_list)
    else:
        return best_cutoff, best_order


def apply_savgol_filter(x, window_length, polyorder):
    return savgol_filter(x, window_length, polyorder)

def save_predictions():
    save_name = 'FINAL'
    if save_latent_training or save_predictions_training:
        base_path_X = "../omg_data/faces_extracted_without_pics/"
        subjects_predictions = [1,2,3,4,5,6,7,8,9,10]
        stories_predictions   = [1,2,4,5,8]
        predictions_n = len(subjects_predictions)*len(stories_predictions)

        if save_latent_training:
            saving_dir_path = "latent_training_" + save_name + "/"
            if not os.path.exists(saving_dir_path):
                os.makedirs(saving_dir_path)
            model_lat = Model(inputs=model.input, outputs=model.get_layer('last_dense').output)
        elif save_predictions_training:
            saving_dir_path = "predictions_training_" + save_name + "/"
            if not os.path.exists(saving_dir_path):
                os.makedirs(saving_dir_path)
            model_lat = model

    elif save_latent_test or save_predictions_test:
        base_path_X = "../omg_data/faces_extracted_test_without_pics/"
        subjects_predictions = [1,2,3,4,5,6,7,8,9,10]
        stories_predictions   = [3,6,7]
        predictions_n = len(subjects_predictions)*len(stories_predictions)

        if save_latent_test:
            saving_dir_path = "latent_test_" + save_name + "/"
            if not os.path.exists(saving_dir_path):
                os.makedirs(saving_dir_path)
            model_lat = Model(inputs=model.input, outputs=model.get_layer('last_dense').output)
        elif save_predictions_test:
            saving_dir_path = "predictions_test_" + save_name + "/"
            if not os.path.exists(saving_dir_path):
                os.makedirs(saving_dir_path)
            model_lat = model


    print(model_lat.summary())

    i = 0
    for subject in subjects_predictions:
        for story in stories_predictions:
            video_name = "Subject_" + str(subject) + "_Story_" + str(story)

            print("\t- Predicting from video: " + video_name + ", " + str(i+1) + "/" + str(predictions_n))

            if save_latent_training or save_predictions_training:
                X_video = np.loadtxt(base_path_X + video_name + ".mp4/Subject_face_landmarks/landmarksSubject.csv",
                                 delimiter=",")
            elif save_latent_test or save_predictions_test:
                X_video = np.loadtxt(base_path_X + video_name + "/Subject_face_landmarks/landmarksSubject.csv",
                                 delimiter=",")

            # preprocessing
            X_video = X_preprocessing(X_video)
            #window sampling
            X_video = X_window_samples(X_video, window_size)

            predictions_lat = model_lat.predict(X_video, verbose=True)

            if save_predictions_training or save_predictions_test:
                predictions_lat = f_trick(Y_training, predictions_lat)[:,np.newaxis]

                np.save(saving_dir_path + video_name + ".npy", predictions_lat)


            print(predictions_lat.shape)
            del X_video

            plt.figure()
            plt.plot(predictions_lat)
            plt.show()

            i +=1
