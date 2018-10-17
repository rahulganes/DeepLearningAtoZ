# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 23:52:06 2018

@author: rg
"""


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 3, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.load_weights('fruit.h5')

from keras.preprocessing import image
import numpy as np

test_image=image.load_img('images.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
print(result)

if result[0][0]==1:
    print("AppleRed")
elif result[0][1]==1:
    print("Banana")
else:
    print("Orange")
