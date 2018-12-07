
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy.stats import pearsonr
from matplotlib import pyplot as plt


def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)

    rho,_ = pearsonr(y_pred,y_true)
    std_predictions = np.std(y_pred)
    std_gt = np.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (
       std_predictions ** 2 + std_gt ** 2 +
       (pred_mean - true_mean) ** 2)

    return ccc, rho



# Fermin's 2018 tricks
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



def get_Y(story, subject, smooth=0):
    file_name = "/Subject_"+str(subject)+"_Story_"+str(story) + ".csv"
    labels_path = "train_val/original_labels" + file_name
    Y = open(labels_path).read().split("\n")[1:-1]
    Y = [float(x) for x in Y]
    return Y


def get_all_Y(stories, subjects, normalize_labels=False, smooth=0):
    Y_list = []
    for subject in subjects:
        for story in stories:
            Y = get_Y(story, subject)
            Y_list.append(Y)
            if smooth>0:
                Y = butter_lowpass_filter_bidirectional(np.array(Y), cutoff=smooth, fs=25, order=1)
            if normalize_labels:
                Y = (Y- np.min(Y))/(np.max(Y)-np.min(Y))

    return np.concatenate(Y_list, axis=0)



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



test_lenghts = np.array([[9025, 5850,7050],
                         [9175,5400,6325],
                         [9450,7000,7000],
                         [8775,4700,5700],
                         [7025,6425,9475],
                         [8850,6500,7850],
                         [8800,5775,8125],
                         [8975,5450,8825],
                         [10325,6100,9550],
                         [10425,5625, 8850]])


# Evaluate with "average prediction" for each subject (WITHOUT filter optimization)
results = []
with_filter = True
subjects = [1,2,3,4,5,6,7,8,9,10]
modalities = ["rawface", "landmarks", "speech", "lexicons", "fullbody"] #["lexicons", "rawface", "landmarks", "speech", "fullbody"]
stories_trainVal = [1,2,4,5,8]
stories_test = [3,6,7]
results_modality = {m:0 for m in modalities}

save_csv = True
save_path = 'test_prediction_FINAL/'

finaldf = pd.DataFrame()
for i, subject in enumerate(subjects):
    for j, story in enumerate(stories_test):
        #model.load_weights(checkpoint_filename)
        #X_val_dic_s = get_all_X(stories_val, [subject], modalities)
        #X_val_list_s = [X_val_dic_s[k] for k in X_val_dic_s]
        Y_trainVal_s = get_all_Y(stories_trainVal, [subject], modalities)
        #Y_test_s = get_all_Y(stories_test, [subject], modalities)
        #Y_val_s


        X_coeff = {
                   "speech":    1. ,
                   "rawface":    .1,
                   "lexicons":  1. ,
                   "landmarks":  .4,
                   "fullbody":  1.
                  }

        filters = {
               "speech":   (0.004,1),
               "rawface":  (0.006,1),
               "lexicons":  (0.01,1),
               "landmarks":(0.004,1),
               "fullbody": (0.004,1)
              }

        X = {}
        len_preds_s = (test_lenghts[i][j])
        preds_s = np.zeros((len_preds_s))
        ourdf = pd.DataFrame({"Subject":np.repeat(subject,len_preds_s)})
        for modality in modalities:
            file_name = "/Subject_"+str(subject)+"_Story_"+str(story)+".npy"
            base_path = "test/"
            latent_vecs_path = base_path + modality + file_name
            X[modality] = np.load(latent_vecs_path)
            print(modality)
            print(X[modality].shape)
            X[modality] = X[modality].flatten()
            X[modality] = butter_lowpass_filter_bidirectional(X[modality], cutoff=filters[modality][0], order=filters[modality][1])
            X[modality] = f_trick(Y_trainVal_s, X[modality])
            X[modality] = X[modality]*X_coeff[modality]
            preds_s += X[modality]
            ourdf[modality]=X[modality]
            finaldf = pd.concat([finaldf,ourdf])


        preds_s /= sum(X_coeff.values())



        if with_filter:
            preds_s = butter_lowpass_filter_bidirectional(preds_s, cutoff=0.01, order=1)
        preds_tricks_s = f_trick(Y_trainVal_s, preds_s)



        plt.figure(figsize=(13, 5))

        for modality in modalities:
            plt.plot(X[modality],label=modality)
        plt.plot(preds_tricks_s,label='average',lw=5)
        plt.legend()
        plt.show()


        if save_csv:
            pd.DataFrame({"valence":preds_tricks_s}).to_csv(save_path+'Subject_{0}_Story_{1}.csv'.format(subject,story))
            pdddd = pd.DataFrame({"valence":preds_tricks_s})
