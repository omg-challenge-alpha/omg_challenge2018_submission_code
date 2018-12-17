## Transcript Model (with Emotional Lexicons)

### dependencies
numpy, keras, tensorflow, sci-kit image, pandas

### preprocessing
The preprocessing.py script takes in input the json files generated with [Amazon Transcribe](https://aws.amazon.com/transcribe/), and generate the features for each word in the dialogs. The features are 11 in total, and are extracted from two emotional lexicons ([1](http://crr.ugent.be/archives/1003), [2](https://github.com/marcoguerini/DepecheMood/releases)). 

### training
The 11 dimensions vectors are then computed as a sequence in the LSTM_lexicons.py script, that takes in input the files generated in the preprocessing phase, and trains an LSTM with an attention module in output. Also the subject are taken in account with a learned embedding of dimension 2 that is concatenated to the LSTM output just before the final affine transformation. 
