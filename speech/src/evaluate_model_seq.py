import numpy as np
import loadconfig
import os
import pandas
import ConfigParser
import essentia.standard as ess
from keras import backend as K
from keras.models import load_model
from scipy.signal import filtfilt, butter
import utilities_func as uf
import utilities_func as uf
from calculateCCC import ccc2
import feat_analysis2 as fa


#load config file
config = loadconfig.load()
cfg = ConfigParser.ConfigParser()
cfg.read(config)

#get values from config file

EVALUATION_PREDICTORS_LOAD = cfg.get('model', 'evaluation_predictors_load')
REFERENCE_PREDICTORS_LOAD = cfg.get('model', 'reference_predictors_load')
EVALUATION_TARGET_LOAD = cfg.get('model', 'evaluation_target_load')
LLD_DIR = cfg.get('model', 'last_latent_dim_dir')
SEQ_LENGTH = cfg.getint('preprocessing', 'sequence_length')
MODEL = cfg.get('model', 'load_model')
SR = cfg.getint('sampling', 'sr')
HOP_SIZE = cfg.getint('stft', 'hop_size')

fps = 25  #annotations per second
hop_annotation = SR /fps
frames_per_annotation = hop_annotation/float(HOP_SIZE)
feats_per_frame = 8
feats_per_valence = int(frames_per_annotation * feats_per_frame)

'''
reminder = frames_per_annotation % 1
if reminder != 0.:
    raise ValueError('Hop size must be a divider of annotation hop (640)')
else:
    frames_per_annotation = int(frames_per_annotation)
'''

#custom loss function
batch_size=50
def batch_CCC(y_true, y_pred):
    CCC = uf.CCC(y_true, y_pred)
    CCC = CCC /float(batch_size)
    return CCC

#load classification model and latent extractor
valence_model = load_model(MODEL, custom_objects={'CCC':uf.CCC,'batch_CCC':batch_CCC})
latent_extractor = K.function(inputs=[valence_model.input], outputs=[valence_model.get_layer('flatten_1').output])

#load datasets rescaling
reference_predictors = np.load(REFERENCE_PREDICTORS_LOAD)
ref_mean = np.mean(reference_predictors)
ref_std = np.std(reference_predictors)
predictors = np.load(EVALUATION_PREDICTORS_LOAD)
target = np.load(EVALUATION_TARGET_LOAD)

print ""
print "using model: " + MODEL


def predict_datapoint(input_sound, input_annotation):
    '''
    loads one audio file and predicts its coutinuous valence

    '''
    sr, samples = uf.wavread(input_sound)  #load
    e_samples = uf.preemphasis(samples, sr)  #apply preemphasis
    predictors = fa.extract_features(e_samples)  #compute power law spectrum
    #normalize by training mean and std
    predictors = np.subtract(predictors, ref_mean)
    predictors = np.divide(predictors, ref_std)
    #load target
    target = pandas.read_csv(input_annotation)
    target = target.values
    target = np.reshape(target,(target.shape[0]))
    final_pred = []
    #compute prediction until last frame
    start = 0
    while start < (len(target)-SEQ_LENGTH):
        start_features = int(start * frames_per_annotation)
        stop_features = int((start + SEQ_LENGTH) * frames_per_annotation)
        predictors_temp = predictors[start_features:stop_features]
        predictors_temp = predictors_temp.reshape(1,predictors_temp.shape[0], predictors_temp.shape[1])
        #predictors_temp = predictors_temp.reshape(1,predictors_temp.shape[0], predictors_temp.shape[1], 1)

        prediction = valence_model.predict(predictors_temp)
        for i in range(prediction.shape[1]):
            final_pred.append(prediction[0][i])
        perc = int(float(start)/(len(target)-SEQ_LENGTH) * 100)
        print "Computing prediction: " + str(perc) + "%"
        start += SEQ_LENGTH
    #compute prediction for last frame
    predictors_temp = predictors[-int(SEQ_LENGTH*frames_per_annotation):]
    predictors_temp = predictors_temp.reshape(1,predictors_temp.shape[0], predictors_temp.shape[1])
    prediction = valence_model.predict(predictors_temp)
    missing_samples = len(target) - len(final_pred)
    #last_prediction = prediction[0][-missing_samples:]
    reverse_index = np.add(list(reversed(range(missing_samples))),1)
    for i in reverse_index:
        final_pred.append(prediction[0][-i])
    final_pred = np.array(final_pred)



    '''
    #compute best prediction shift
    shifted_cccs = []
    time = np.add(1,range(200))
    print "Computing best optimization parameters"
    for i in time:
        t = target.copy()
        p = final_pred.copy()
        t = t[i:]
        p = p[:-i]
        #print t.shape, p.shape

        temp_ccc = ccc2(t, p)
        shifted_cccs.append(temp_ccc)


    best_shift = np.argmax(shifted_cccs)
    best_ccc = np.max(shifted_cccs)
    if best_shift > 0:
        best_target = target[best_shift:]
        best_pred = final_pred[:-best_shift]
    else:
        best_target = target
        best_pred = final_pred
    #print 'LEN BEST PRED: ' + str(len(best_pred))

    #compute best parameters for the filter
    test_freqs = []
    test_orders = []
    test_cccs = []
    freqs = np.arange(0.01,0.95,0.01)
    orders = np.arange(1,10,1)
    print "Finding best optimization parameters..."
    for freq in freqs:
        for order in orders:
            test_signal = best_pred.copy()
            b, a = butter(order, freq, 'low')
            filtered = filtfilt(b, a, test_signal)
            temp_ccc = ccc2(best_target, filtered)
            test_freqs.append(freq)
            test_orders.append(order)
            test_cccs.append(temp_ccc)
    best_filter = np.argmax(test_cccs)
    best_order = test_orders[best_filter]
    best_freq = test_freqs[best_filter]
    '''
    #POSTPROCESSING
    #normalize between -1 and 1
    final_pred = np.multiply(final_pred, 2.)
    final_pred = np.subtract(final_pred, 1.)

    #apply f_trick
    ann_folder = '../dataset/Training/Annotations'
    target_mean, target_std = uf.find_mean_std(ann_folder)
    final_pred = uf.f_trick(final_pred, target_mean, target_std)

    #apply butterworth filter
    b, a = butter(3, 0.01, 'low')
    final_pred = filtfilt(b, a, final_pred)

    ccc = ccc2(final_pred, target)  #compute ccc
    print "CCC = " + str(ccc)

    '''
    plt.plot(target)
    plt.plot(final_pred, alpha=0.7)
    plt.legend(['target','prediction'])
    plt.show()
    '''

    return ccc

def extract_LLD_datapoint(input_sound, input_annotation):
    '''
    load one audio file and compute the model's last
    latent dimension
    '''
    sr, samples = uf.wavread(input_sound)  #load
    e_samples = uf.preemphasis(samples, sr)  #apply preemphasis
    predictors = fa.extract_features(e_samples)  #compute power law spectrum
    #normalize by training mean and std
    predictors = np.subtract(predictors, ref_mean)
    predictors = np.divide(predictors, ref_std)
    final_vec = np.array([])
    #load target
    target = pandas.read_csv(input_annotation)
    target = target.values
    target = np.reshape(target,(target.shape[0]))

    #compute last latent dim until last frame
    start = 0
    while start < (len(target)-SEQ_LENGTH):
        start_features = int(start * frames_per_annotation)
        stop_features = int((start + SEQ_LENGTH) * frames_per_annotation)
        predictors_temp = predictors[start_features:stop_features]
        predictors_temp = predictors_temp.reshape(1,predictors_temp.shape[0], predictors_temp.shape[1])
        features_temp = latent_extractor([predictors_temp])
        features_temp = np.reshape(features_temp, (SEQ_LENGTH, feats_per_valence))
        if final_vec.shape[0] == 0:
            final_vec = features_temp
        else:
            final_vec = np.concatenate((final_vec, features_temp), axis=0)
        print 'Progress: '+ str(int(100*(final_vec.shape[0] / float(len(target))))) + '%'
        start += SEQ_LENGTH
    #compute last latent dim for last frame
    predictors_temp = predictors[-int(SEQ_LENGTH*frames_per_annotation):]
    predictors_temp = predictors_temp.reshape(1,predictors_temp.shape[0], predictors_temp.shape[1])
    features_temp = latent_extractor([predictors_temp])
    features_temp = np.reshape(features_temp, (SEQ_LENGTH, feats_per_valence))
    missing_samples = len(target) - final_vec.shape[0]
    last_vec = features_temp[-missing_samples:]
    final_vec = np.concatenate((final_vec, last_vec), axis=0)

    return final_vec

def evaluate_all_data(sound_dir, annotation_dir):
    '''
    compute prediction and ccc for all validation set
    '''
    list = os.listdir(annotation_dir)
    list = list[:]
    ccc = []
    for datapoint in list:
        annotation_file = annotation_dir + '/' + datapoint
        name = datapoint.split('.')[0]
        print 'Processing: ' + name
        sound_file = sound_dir + '/' + name +".mp4.wav"
        temp_ccc = predict_datapoint(sound_file, annotation_file)
        ccc.append(temp_ccc)
    ccc = np.array(ccc)
    mean_ccc = np.mean(ccc)
    min_ccc = np.min(ccc)
    max_ccc = np.max(ccc)

    print "Mean CCC = " + str(mean_ccc)
    print "Min CCC = " + str(min_ccc)
    print "Max CCC = " + str(max_ccc)

def extract_LLD_dataset(sound_dir, annotation_dir):
    '''
    compute last latent dimension for all dataset
    '''
    list = os.listdir(annotation_dir)
    list = list[:]
    for datapoint in list:
        annotation_file = annotation_dir + '/' + datapoint
        name = datapoint.split('.')[0]
        print 'Processing: ' + name
        sound_file = sound_dir + '/' + name +".mp4.wav"
        lld = extract_LLD_datapoint(sound_file, annotation_file)
        output_filename = LLD_DIR + '/' + name + '.npy'
        np.save(output_filename, lld)
