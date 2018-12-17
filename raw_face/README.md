## Raw Face Model

The raw face model predicts valence from sequences of subject face cropped frames using a 3d convolutional architecture. This model additionally takes into account the identity of the subject as an auxiliary input added after the convolutional layers.

### dependencies

numpy, keras, tensorflow, sci-kit image 
 

### preprocessing

The preprocessing of the videos is done using the script omg_challenge2018_submission_code/landmarks/landmarks_preprocessing.py.
This operation crops the face of the subject at every frame and transforms it into an equalized 48x48 BW image.


### training

To load the dataset and train the model simply call raw_face_main.py (make sure you provide the correct path to load the images). This script will save in the target folder the best model given the validation performance.

