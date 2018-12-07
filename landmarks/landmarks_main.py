
from model import *
from utils import *

import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr


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

import time
#from time import time

base_path_X = "../omg_data/faces_extracted_without_pics/"
base_path_Y = "../omg_data/Annotations/"

subjects_training = [1,2,3,4,5,6,7,8,9,10]
subjects_validation = [1,2,3,4,5,6,7,8,9,10]
subjects_test = [1,2,3,4,5,6,7,8,9,10]

stories_training   = [1,4,5,8]
stories_validation = [2]
stories_test = [3,6,7]

subject_data = True
actor_data   = False


if subject_data == actor_data:
    raise Exception("Choose between subject and actor data")

window_size = 5
X_training, Y_training, X_validation, Y_validation, indexes_training, indexes_validation = load_dataset(window_size)

day_time = time.strftime("%Y-%m-%d_%H_%M_%S")
experiment_name = "experiment_" + day_time
os.mkdir(experiment_name)

filepath=experiment_name + "/"


batch_size = 512
epochs = 1000
patience = 1000
lr = 0.000001

opt = optimizers.Adam(lr=lr, decay=0.0)

N_features = X_training.shape[2]

model = build_model(window_size, N_features)
#model = build_model_LSTM()

print(model.summary())

model.compile(loss=ccc_error, #ccc_error #mean_squared_error #pearson_error
              optimizer=opt)

metrics_callback = Metrics()
csv_logger = CSVLogger('training.log')

callbacks_list = [  #csv_logger,
                    metrics_callback,
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
                    keras.callbacks.TensorBoard(log_dir="logs/" + experiment_name),
                    #keras.callbacks.ModelCheckpoint(filepath=experiment_name+'/weights_epoch_{epoch:02d}.h5', monitor='val_loss', save_best_only=False)
                 ]

with open(experiment_name+'/modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

#pp= 8193
#ppp = 3000

history = model.fit(X_training, Y_training,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_validation, Y_validation),
                    callbacks=callbacks_list)

#metrics_callback.get_data()

save_latent_training = False
save_predictions_training = False
save_latent_test = False
save_predictions_test = True

model.load_weights('best_model.h5')
save_predictions()
