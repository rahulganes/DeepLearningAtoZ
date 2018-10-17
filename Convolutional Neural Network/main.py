
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

#Step 1:Convolutional Layer - form a feature map
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Step 2:MaxPooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#step 3:Flattening layer - form a vector
classifier.add(Flatten())

#step 4:full Connection layer 
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 3, activation = 'softmax'))

#step 5:Complile the neuralnetwork
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Preprocessing the Image dataset
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (64, 64),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size = (64, 64),
                                            batch_size = 10,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         samples_per_epoch = 470,
                         nb_epoch = 50,
                         steps_per_epoch = 470,
                         validation_data = test_set,
                         nb_val_samples = 150)

classifier.save_weights('fruit.h5')
