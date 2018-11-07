# Convolutional neural network from venkateshtata with simple modifications to use it more easily
A simple binary classifier to predict if the given image is a cat or a dog.

# Installing Tensorflow
pip install tensorflow

# Installing Keras
pip install --upgrade keras

# Download the test and training images and put in a folder called dataset
https://drive.google.com/drive/folders/1XaFM8BJFligrqeQdE-_5Id0V_SubJAZe
Create a folder inside dataset called single_prediction and put a image of a dog or a cat of your preference

Run the trainCNN.py

Pay atention to this part of code, change this to increase acurancy.
steps_per_epoch = 2000,
epochs = 3,
validation_steps = 1000)

After that you need to run runCNN.py, this script will use the model.h5 to make his prediction
