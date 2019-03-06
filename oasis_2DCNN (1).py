#!/usr/bin/env python
# coding: utf-8

# In[11]:



from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import optimizers
import nilearn


# In[20]:


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


# In[63]:


import numpy as np
import pandas as pd
import zipfile
from keras.preprocessing.image import ImageDataGenerator 


# In[57]:


# download oasis dataset on working directory 
from nilearn import datasets
from nilearn.input_data import NiftiMasker

from nilearn.image import smooth_img
import keras

oasis_dataset = datasets.fetch_oasis_vbm(n_subjects= 416)


# In[13]:


#gm_imgs = np.array(oasis.gray_matter_maps)
gray_matter_map_filenames = oasis_dataset.gray_matter_maps
gm_imgs = gray_matter_map_filenames


# In[16]:


# create binary label by clinical dimentia rating (CDR)    
cdr = oasis_dataset.ext_vars['cdr'].astype(float)
cdr_numpy_arr = np.array(cdr)
for i in range(len(cdr_numpy_arr)):
    if(np.isnan(cdr_numpy_arr[i])): cdr_numpy_arr[i] = 1
    
    elif(cdr_numpy_arr[i] > 0.0): cdr_numpy_arr[i] = 1


# In[16]:


pwd


# In[ ]:





# In[26]:


train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('NiDataTrain',
target_size = (64, 64),
class_mode = 'binary')


# In[25]:


test_set = test_datagen.flow_from_directory('NiDataTest',
target_size = (64, 64),
class_mode = 'binary')


# In[52]:


test_set.class_indices


# In[62]:



test_labels = test_set.classes
train_labels = training_set.classes
testy = pd.DataFrame(test_labels)


# In[18]:


import keras
from keras.utils import to_categorical

label = keras.utils.to_categorical(cdr_numpy_arr, 2)


# In[21]:


classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000,
                        workers=4
                        )


# In[28]:


classifier.save_weights('25_epochs.h5')


# In[37]:


classifier.save_weights('25_epochs.csv')


# In[41]:


# test loss and accuracy 
classifier.evaluate_generator(generator=test_set, steps = 20)


# In[44]:


# train loss and accuracy 
classifier.evaluate_generator(generator= training_set, steps = 20)
# this is overfitting because the training accuracy is 100% and test is 60% 


# In[66]:


STEP_SIZE_TRAIN =training_set.n//training_set.batch_size
#STEP_SIZE_VALID =valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST =test_set.n//test_set.batch_size
#model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_generator,validation_steps=STEP_SIZE_VALID, epochs=10


# In[67]:


test_set.reset()
pred=classifier.predict_generator(test_set,
steps=STEP_SIZE_TEST,
verbose=1)


# In[69]:


predicted_class_indices=np.argmax(pred,axis=1)


# In[72]:


labels = (test_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
predictions


# In[ ]:




