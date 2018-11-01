import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.models import load_model

classifier = Sequential()
classifier = load_model('model.h5')
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print ('The image has a...')
print (prediction)
