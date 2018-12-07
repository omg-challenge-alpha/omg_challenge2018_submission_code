import numpy as np
import math, copy
import os
import pandas
from scipy.io.wavfile import read, write
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt

tol = 1e-14    # threshold used to compute phase

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

def isPower2(num):
    #taken from Xavier Serra's sms tools
    """
    Check if num is power of two
    """
    return ((num & (num - 1)) == 0) and num > 0

def wavread(file_name):
    #taken from Xavier Serra's sms tools
    '''
    read wav file and converts it from int16 to float32
    '''
    sr, samples = read(file_name)
    samples = np.float32(samples)/norm_fact[samples.dtype.name] #float conversion

    return sr, samples

def wavwrite(y, fs, filename):
    #taken from Xavier Serra's sms tools
    """
    Write a sound file from an array with the sound and the sampling rate
    y: floating point array of one dimension, fs: sampling rate
    filename: name of file to create
    """
    x = copy.deepcopy(y)                         # copy array
    x *= INT16_FAC                               # scaling floating point -1 to 1 range signal to int16 range
    x = np.int16(x)                              # converting to int16 type
    write(filename, fs, x)

def dftAnal(x, w, N):
    #taken from Xavier Serra's sms tools
	"""
	Analysis of a signal using the discrete Fourier transform
	x: input signal, w: analysis window, N: FFT size
	returns mX, pX: magnitude and phase spectrum
	"""

	if not(isPower2(N)):                                 # raise error if N not a power of two
		raise ValueError("FFT size (N) is not a power of 2")

	if (w.size > N):                                        # raise error if window size bigger than fft size
		raise ValueError("Window size (M) is bigger than FFT size")

	hN = (N/2)+1                                            # size of positive spectrum, it includes sample 0
	hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
	fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
	w = w / sum(w)                                          # normalize analysis window
	xw = x*w                                                # window the input sound
	fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
	fftbuffer[-hM2:] = xw[:hM2]
	X = fft(fftbuffer)                                      # compute FFT
	absX = abs(X[:hN])                                      # compute ansolute value of positive side
	absX[absX<np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
	mX = 20 * np.log10(absX)                                # magnitude spectrum of positive frequencies in dB
	X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0            # for phase calculation set to 0 the small values
	X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0            # for phase calculation set to 0 the small values
	pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies

	return mX, pX

def stftAnal(x, w, N, H) :
    #taken from Xavier Serra's sms tools
	"""
	Analysis of a sound using the short-time Fourier transform
	x: input array sound, w: analysis window, N: FFT size, H: hop size
	returns xmX, xpX: magnitude and phase spectra
	"""
	if (H <= 0):                                   # raise error if hop size 0 or negative
		raise ValueError("Hop size (H) smaller or equal to 0")

	M = w.size                                      # size of analysis window
	hM1 = int(math.floor((M+1)/2))                  # half analysis window size by rounding
	hM2 = int(math.floor(M/2))                      # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                  # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM2))                  # add zeros at the end to analyze last sample
	pin = hM1                                       # initialize sound pointer in middle of analysis window
	pend = x.size-hM1                               # last sample to start a frame
	w = w / sum(w)                                  # normalize analysis window
	while pin<=pend:                                # while sound pointer is smaller than last sample
		x1 = x[pin-hM1:pin+hM2]                       # select one frame of input sound
		mX, pX = dftAnal(x1, w, N)                # compute dft
		if pin == hM1:                                # if first frame create output arrays
			xmX = np.array([mX])
			xpX = np.array([pX])
		else:                                         # append output to existing array
			xmX = np.vstack((xmX,np.array([mX])))
			xpX = np.vstack((xpX,np.array([pX])))
		pin += H                                      # advance sound pointer

	return xmX, xpX


def preemphasis(input_vector, fs):
    '''
    2 simple high pass FIR filters in cascade to emphasize high frequencies
    and cut unwanted low-frequencies
    '''
    #first gentle high pass
    alpha=0.5
    present = input_vector
    zero = [0]
    past = input_vector[:-1]
    past = np.concatenate([zero,past])
    past = np.multiply(past, alpha)
    filtered1 = np.subtract(present,past)
    #second 30 hz high pass
    fc = 100.  # Cut-off frequency of the filter
    w = fc / (fs / 2.) # Normalize the frequency
    b, a = butter(8, w, 'high')
    output = filtfilt(b, a, filtered1)

    return output

def CCC(y_true, y_pred):
    '''
    Lin's Concordance correlation coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Accepting tensors as input

    '''

    import keras.backend as K
    # covariance between y_true and y_pred
    N = K.int_shape(y_pred)[-1]
    s_xy = 1.0 / (N - 1.0 + K.epsilon()) * K.sum((y_true - K.mean(y_true)) * (y_pred - K.mean(y_pred)))
    # means
    x_m = K.mean(y_true)
    y_m = K.mean(y_pred)
    # variances
    s_x_sq = K.var(y_true)
    s_y_sq = K.var(y_pred)

    # condordance correlation coefficient
    ccc = (2.0*s_xy) / (s_x_sq + s_y_sq + (x_m-y_m)**2)

    return ccc


def find_mean_std(input_folder):
    annotations = os.listdir(input_folder)
    sequence = []
    for datapoint in annotations:
        annotation_file = input_folder + '/' + datapoint
        ann = pandas.read_csv(annotation_file)
        ann = ann.values.T[0]
        sequence = np.concatenate((sequence,ann))
    mean = np.mean(sequence)
    std = np.std(sequence)
    return mean, std

def f_trick(input_sequence, ref_mean, ref_std):
    mean = np.mean(input_sequence)
    std = np.std(input_sequence)
    num = np.multiply(ref_std, np.subtract(input_sequence, mean))
    output = np.divide(num, std)
    output = np.add(output, ref_mean)

    return output

def gen_fake_annotations(frames_count, output_folder):
    with open(frames_count) as f:
        content = f.readlines()
    for line in content:
        split = line.split('-')
        name = split[0].replace(' ', '')
        file_name = output_folder + '/' + name
        len = int(split[-1].split(' ')[1])
        valence = np.zeros(len)
        temp_dict = {'valence':valence}
        temp_df = pandas.DataFrame(data=temp_dict)
        temp_df.to_csv(file_name, index=False)
