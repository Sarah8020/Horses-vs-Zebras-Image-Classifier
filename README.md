# Horses-vs-Zebras-Image-Classifier
Simple image classifier that differentiates between images of horses and zebras using deep learning.

The model includes data augmentation to introduce artificial diversity to the training set. Also included is code to run inference on data that was not included in the training set. This data includes four pictures, two of horses and two of zebras. The images are loaded into PIL format using the load_img utility and then used the img_to_array utility to convert the PIL Image to a Numpy array. expand_dims is used to add a batch dimension so that the image array can be accepted by the model. The predict function is then used to generate predictions, and then those predictions are interpreted & printed accordingly.

The number of training epochs was set to 8 to keep the overall training time around 5 minutes (may vary). After 8 training epochs, the model typically achieves a validation accuracy >85%.

Instructions for training the model:
1) Requires TensorFlow 2 and Python 3.7-3.9.
2) Download all files from the Github repository
3) Download this Kaggle dataset: www.kaggle.com/balraj98/horse2zebra-dataset
4) From the downloaded files, use only the files in the training subset ('trainA' & 'trainB'). Include those folders under the 'data' directory, but rename the folders 'horses' and 'zebras' respectively.
5) Remove the 4 images specified in the 'test' directory from the 'data' directory - these 4 images used for testing were pulled from the training data, and should not be included in the training data used to train the model in order to achieve accurate inference results.
6) Open the project.py file in your IDE of choice and run it. Or, after navigating to the proper directory, execute with the following command: python3 project.py
