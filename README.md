# OMG Empathy Challenge 2019
# Alpha - City Team

[Telefónica Innovation Alpha](https://www.alpha.company/)
[MIRG - City, University of London](https://mirg.city.ac.uk/)


This is an ensemble multimodal model that integrates predictions from different sources (video, audio, and dialogue transcript). To run the full model, each individual module needs to be run separately and the prediction of each of them integrated using one of the proposed methods (Regression model, Smoothed weighted average model or K-nearest Neighbours Model).


## Raw Face Model

The raw face model predicts valence from sequences of subject face cropped frames using a 3d convolutional architecture. This model additionally takes into account the identity of the subject as an auxiliary input added after the convolutional layers.

### dependencies

numpy, keras, tensorflow, sci-kit image 
 

### preprocessing

The preprocessing of the videos is done using the script preprocessing/faces_and_landmarks_extraction/extract_faces_and_landmarks.py
This operation crops the face of the subject at every frame and transforms it into an equalized 48x48 BW image.


### training

To load the dataset and train the model simply call raw_face_main.py (make sure you provide the correct path to load the images). This script will save in the target folder the best model given the validation performance.


## Facial Landmarks model

### dependencies
numpy, keras, tensorflow, sci-kit image, pandas, scipy

### preprocessing
The preprocessing of the videos is done using the script preprocessing/faces_and_landmarks_extraction/extract_faces_and_landmarks.py
This operation detects 68 face landmarks in each frame for all the videos and transform them in a csv file. The preprocessing script needs the data file shape_predictor_68_face_landmarks.dat from the dlib library, downloadable from github.

### training
To load the dataset and train the model call landmarks_main.py (make sure you provide the correct path to load the landmarks file from the preprocessing). This script will save in the target folder the best model given the validation performance.

## Full Body Model

The full body model predicts valence from sequences of subject full body cropped frames using a 3d convolutional architecture based on Resnet 16.

### dependencies

numpy, keras, tensorflow, sci-kit image, [keras-resnet3d](https://github.com/JihongJu/keras-resnet3d)

### preprocessing

The preprocessing of the videos is done using the script preprocessing/full_body_extraction/full_body.py
This operation crops the full body image of the subject at every frame and transforms it into an equalized 128x128 grayscale image.

### training

To load the dataset and train the model call fullbody_main.py (make sure you provide the correct path to load the images). This script will save in the target folder the best model given the validation performance.


## Transcript Model (with Emotional Lexicons)

### dependencies
numpy, keras, tensorflow, sci-kit image, pandas

### preprocessing
The preprocessing.py script takes in input the json files generated with [Amazon Transcribe](https://aws.amazon.com/transcribe/), and generate the features for each word in the dialogs. The features are 11 in total, and are extracted from two emotional lexicons ([1](), [2](https://github.com/marcoguerini/DepecheMood/releases)). 

### training
The 11 dimensions vectors are then computed as a sequence in the LSTM_lexicons.py script, that takes in input the files generated in the preprocessing phase, and trains an LSTM with an attention module in output. Also the subject are taken in account with a learned embedding of dimension 2 that is concatenated to the LSTM output just before the final affine transformation. 

## Multimodal Regression Model

The multimodal regression model integrates the results of the different streams into a single prediction.

### Dependencies
-	pandas
-	numpy
-	scipy
-	statsmodels

### Contents
-	The pickled model files (a PCA and a linear regression model)
-	A pickled filed with per subject means and standard deviations for rescaling
-	A CSV files with the test data preformatted (with the outputs of the other models, already rescaled and filtered)
-	The python code to achieve the predictions.

### Model description
Prior to applying the PCA and regression, the results of the modality-specific models are lowpass filtered using Butterworth filters, and then rescaled to the desired per subject mean and standard deviation.

### To run
With all files in the same directory, load and run the python file. It will produce two Panda’s DataFrames with the valence in the last column (currently, for illustration it just prints the lines of the final resulting DataFrames)

## Multimodal 5-NN Model

The multimodal 5-Nearest Neighbors model integrates the results of the different streams into a single prediction.

### Dependencies
-	pandas
-	numpy
-	scipy
-	sklearn

### Contents
-	The python code to train, cross-validate, achieve the predictions.

### Model description
A 5-NN regression model whose inputs are the 5-dimensional embeddings of the the 5 modalities (with a lag of 5 timeframes between embedding points). After applying the algorithm, the outputs are lowpass filtered and rescaled to the specific subject.


## Multimodal Manual Smoothed Weighted Average Model

The Manual Smoothed Weighted Average Model integrates the results of the different streams into a single prediction.

### Dependencies
-	pandas
-	numpy
-	scipy

### To run
With all files in the specified directory, load and run the file average_predictions_manual/average_predictions_manual.py. It will produce a Panda DataFrames with the valence in the last column.


