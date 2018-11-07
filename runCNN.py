import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.models import load_model

classifier = Sequential()
classifier = load_model('model.h5')
test_array = [image.load_img('dataset/single_prediction/1.jpg', target_size = (64,64)),
			  image.load_img('dataset/single_prediction/2.jpg', target_size = (64,64)),
			  image.load_img('dataset/single_prediction/3.jpg', target_size = (64,64)),
			  image.load_img('dataset/single_prediction/4.jpg', target_size = (64,64)),
			  image.load_img('dataset/single_prediction/5.jpg', target_size = (64,64))]

for x in range(0,5):
	test_array[x] = image.img_to_array(test_array[x])

for x in range(0,5):
	test_array[x] = np.expand_dims(test_array[x], axis = 0)

for x in range(0,5):
	result = classifier.predict(test_array[x])
	print('Result ', x, 'output ', result[0][0])
	if result[0][0] == 1:
		print('dog')
	else:
		print('cat')
