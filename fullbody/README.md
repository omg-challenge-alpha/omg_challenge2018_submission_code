## Full Body Model

The full body model predicts valence from sequences of subject full body cropped frames using a 3d convolutional architecture based on Resnet 16.

### dependencies

numpy, keras, tensorflow, sci-kit image, [keras-resnet3d](https://github.com/JihongJu/keras-resnet3d)

### preprocessing

The preprocessing of the videos is done using the script omg_challenge2018_submission_code/fullbody/fullbody_preprocessing.py.
This operation crops the full body image of the subject at every frame and transforms it into an equalized 128x128 grayscale image.

### training

To load the dataset and train the model call fullbody_main.py (make sure you provide the correct path to load the images). This script will save in the target folder the best model given the validation performance.
