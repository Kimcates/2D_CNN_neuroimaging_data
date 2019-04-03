import time

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            Name = "{}-conv-{}-nodes'{}-dense{}".format(conv_layer, layer_size, dense_layer, int(time.time()))

            classifier = Sequential()
            # convolution layer: weighted sum between two signals. Features are extracted at k x k sized matrices to calculate the convolution at a specific x, y location
            classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation = 'relu'))
            # Pooling
            classifier.add(MaxPooling2D(pool_size = (2, 2)))
            classifier.add(Dropout(0.25))

            # Add convolution layer 2
            classifier.add(Conv2D(32,3,3,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2,2)))
            classifier.add(Dropout(0.25))
            # Flatten
            classifier.add(Flatten())

            # Full connection
            classifier.add(Dense(units = 128, activation = 'relu'))
            classifier.add(Dropout(0.4))
            classifier.add(Dense(units = 1, activation = 'sigmoid'))

            # compile
            classifier.compile(loss= 'binary_crossentropy', optimizer = 'Adam',metrics=['accuracy'])
