#CONVOLUTIONAL NEURAL NETWORK
#tuned as in https://www.researchgate.net/publication/306187492_Deep_Convolutional_Neural_Networks_and_Data_Augmentation_for_Environmental_Sound_Classification

import numpy as np
from keras.models import Model
from keras.layers import Input, GRU, Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, Reshape, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras import optimizers
from keras import regularizers
import utilities_func as uf
import loadconfig
import ConfigParser
import matplotlib.pyplot as plt
np.random.seed(1)

print "loading dataset..."
config = loadconfig.load()
cfg = ConfigParser.ConfigParser()
cfg.read(config)

#load parameters from config file
NEW_CONV_MODEL = cfg.get('model', 'save_model')
TRAINING_PREDICTORS = cfg.get('model', 'training_predictors_load')
TRAINING_TARGET = cfg.get('model', 'training_target_load')
VALIDATION_PREDICTORS = cfg.get('model', 'validation_predictors_load')
VALIDATION_TARGET = cfg.get('model', 'validation_target_load')
SEQ_LENGTH = cfg.getint('preprocessing', 'sequence_length')
print "Training predictors: " + TRAINING_PREDICTORS
print "Training target: " + TRAINING_TARGET
print "Validation predictors: " + VALIDATION_PREDICTORS
print "Validation target: " + VALIDATION_TARGET

#load datasets
training_predictors = np.load(TRAINING_PREDICTORS)
training_target = np.load(TRAINING_TARGET)
validation_predictors = np.load(VALIDATION_PREDICTORS)
validation_target = np.load(VALIDATION_TARGET)

#rescale datasets to mean 0 and std 1 (validation with respect
#to training mean and std)
tr_mean = np.mean(training_predictors)
tr_std = np.std(training_predictors)
v_mean = np.mean(validation_predictors)
v_std = np.std(validation_predictors)
training_predictors = np.subtract(training_predictors, tr_mean)
training_predictors = np.divide(training_predictors, tr_std)
validation_predictors = np.subtract(validation_predictors, tr_mean)
validation_predictors = np.divide(validation_predictors, tr_std)

#normalize target between 0 and 1
training_target = np.multiply(training_target, 0.5)
training_target = np.add(training_target, 0.5)
validation_target = np.multiply(validation_target, 0.5)
validation_target = np.add(validation_target, 0.5)

#hyperparameters
batch_size = 100
num_epochs = 200
lstm1_depth = 250
hidden_size = 8
drop_prob = 0.3
dense_size = 100
regularization_lambda = 0.01

reg = regularizers.l2(regularization_lambda)
sgd = optimizers.SGD(lr=0.001, decay=0.003, momentum=0.5)
opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#custom loss
def batch_CCC(y_true, y_pred):
    CCC = uf.CCC(y_true, y_pred)
    CCC = CCC /float(batch_size)
    CCC = 1-CCC
    return CCC

time_dim = training_predictors.shape[1]
features_dim = training_predictors.shape[2]

#callbacks
best_model = ModelCheckpoint(NEW_CONV_MODEL, monitor='val_loss', save_best_only=True, mode='min')  #save the best model
early_stopping_monitor = EarlyStopping(patience=5)  #stop training when the model is not improving
callbacks_list = [early_stopping_monitor, best_model]

#model definition
input_data = Input(shape=(time_dim, features_dim))
gru = Bidirectional(GRU(lstm1_depth, return_sequences=True))(input_data)
norm = BatchNormalization()(gru)
hidden = TimeDistributed(Dense(hidden_size, activation='linear'))(norm)
drop = Dropout(drop_prob)(hidden)
flat = Flatten()(drop)
out = Dense(SEQ_LENGTH, activation='linear')(flat)

#model creation
valence_model = Model(inputs=input_data, outputs=out)
#valence_model.compile(loss=batch_CCC, optimizer=opt)
valence_model.compile(loss=batch_CCC, optimizer=opt)

print valence_model.summary()

#model training
history = valence_model.fit(training_predictors, training_target, epochs = num_epochs, validation_data=(validation_predictors,validation_target), callbacks=callbacks_list, batch_size=batch_size, shuffle=True)

print "Train loss = " + str(min(history.history['loss']))
print "Validation loss = " + str(min(history.history['val_loss']))


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MODEL PERFORMANCE', size = 15)
plt.ylabel('loss', size = 15)
plt.xlabel('Epoch', size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend(['train', 'validation'], fontsize = 12)

plt.show()
