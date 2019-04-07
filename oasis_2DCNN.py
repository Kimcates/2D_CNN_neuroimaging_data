#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import zipfile
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import nilearn
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.image import smooth_img
import nibabel as nib
import matplotlib.pyplot as plt
import pickle

classifier = Sequential()
# convolution layer: weighted sum between two signals. Features are extracted at k x k sized matrices to calculate the convolution at a specific x, y location
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Add convolution layer 2
classifier.add(Conv2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
# Flatten
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# compile
classifier.compile(loss= 'binary_crossentropy', optimizer = 'Adam',metrics=['accuracy'])

#Load the training data
pickle_in = open ("gm_imgs_train", "rb")
gm_imgs_train = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open ("labels_train", "rb")
labels_train = pickle.load(pickle_in)
pickle_in.close()

#normalize the training data
gm_imgs_train_normalized = gm_imgs_train/255
#fit the model with training data
classifier.fit(gm_imgs_train, labels_train, validation_split=0.1)









"""
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('NiDataTrain',
target_size = (64, 64),
class_mode = 'binary')

test_set = test_datagen.flow_from_directory('NiDataTest',
target_size = (64, 64),
class_mode = 'binary')


test_set.class_indices



test_labels = test_set.classes
train_labels = training_set.classes
testy = pd.DataFrame(test_labels)

label = keras.utils.to_categorical(cdr_numpy_arr, 2)


classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000,
                        workers=4
                        )


classifier.save_weights('25_epochs.h5')

classifier.save_weights('25_epochs.csv')

classifier.evaluate_generator(generator=test_set, steps = 20)

# train loss and accuracy
classifier.evaluate_generator(generator= training_set, steps = 20)
# this is overfitting because the training accuracy is 100% and test is 60%


STEP_SIZE_TRAIN =training_set.n//training_set.batch_size
#STEP_SIZE_VALID =valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST =test_set.n//test_set.batch_size
#model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_generator,validation_steps=STEP_SIZE_VALID, epochs=10

test_set.reset()
pred=classifier.predict_generator(test_set,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (test_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
predictions
"""
