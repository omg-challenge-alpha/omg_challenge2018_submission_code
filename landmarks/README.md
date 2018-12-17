## Facial Landmarks model

### dependencies
numpy, keras, tensorflow, sci-kit image, pandas, scipy

### preprocessing
The preprocessing of the videos is done using the script omg_challenge2018_submission_code/landmarks/landmarks_preprocessing.py
This operation detects 68 face landmarks in each frame for all the videos and transform them in a csv file. The preprocessing script needs the data file shape_predictor_68_face_landmarks.dat from the dlib library, downloadable from github.

### training
To load the dataset and train the model call landmarks_main.py (make sure you provide the correct path to load the landmarks file from the preprocessing). This script will save in the target folder the best model given the validation performance.
