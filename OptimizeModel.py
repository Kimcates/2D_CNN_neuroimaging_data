import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pickle
import time
from keras.callbacks import TensorBoard

#Load the training data
pickle_in = open ("gm_imgs_train", "rb")
gm_imgs_train = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open ("labels_train", "rb")
labels_train = pickle.load(pickle_in)
pickle_in.close()

#normalize the training data
gm_imgs_train_normalized = gm_imgs_train/255

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            name = "{}-conv-{}-nodes'{}-dense{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir = 'logs/{}'.format(name))
            print(name)

            classifier = Sequential()
            # convolution layer: weighted sum between two signals. Features are extracted at k x k sized matrices to calculate the convolution at a specific x, y location
            classifier.add(Conv2D(layer_size, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
            # Pooling
            classifier.add(MaxPooling2D(pool_size = (2, 2)))

            for layer in range(conv_layer - 1):
                classifier.add(Conv2D(layer_size,3,3, activation='relu'))
                classifier.add(MaxPooling2D(pool_size=(2,2)))

            # Flatten
            classifier.add(Flatten())

            for layer in range(dense_layer):
                # Full connection
                classifier.add(Dense(units = layer_size, activation = 'relu'))

            classifier.add(Dense(units = 1, activation = 'sigmoid'))
            # compile
            classifier.compile(loss= 'binary_crossentropy', optimizer = 'Adam',metrics=['accuracy'])
            #fit the model with training data
            classifier.fit(gm_imgs_train, labels_train, epochs = 10, validation_split=0.1, callbacks=[tensorboard])
