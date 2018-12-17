## Speech Model

This model is based on the audio information extracted from the video data. It has a sequence to sequence design based on a Recurrent Neural Network. The model's task is to predict time sequences of 200 valence samples for 8 seconds of input.
The I/O folders and several pre-processing parameters are set in the config.ini file. Make sure to input the correct paths in this file before launching the sctripts.

### dependencies

numpy, essentia, scipy, configparser, keras, tensorflow, pandas

### preprocessing

The preprocessing.py script applies 4 concatenated transformations to the audio signals: pre-emphasis, segmentation, spectrum computation, power-law compression. This operation produces data-points shaped as (800, 129), where the first dimension is the time and the second is the frequency.

### training

The script build_model_rnn_seq.py first normalizes the input data in order to obtain a dataset with mean equal to 0 and standard deviation equal to 1. Then, it trains the RNN-based model and saves the best one with respect to the lowest validation loss.