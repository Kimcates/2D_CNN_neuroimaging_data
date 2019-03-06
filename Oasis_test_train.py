#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Preliminaries 
import numpy as np
import matplotlib.pyplot as plt 
import scipy
from scipy import linalg

from nilearn import datasets
from nilearn.input_data import NiftiMasker

from nilearn.image import smooth_img
import numpy as np
#import cv2

import keras


# In[2]:


# download oasis dataset on working directory 
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects= 416)


# In[5]:


#gm_imgs = np.array(oasis.gray_matter_maps)
gray_matter_map_filenames = oasis_dataset.gray_matter_maps
gm_imgs = gray_matter_map_filenames


# In[6]:


# create binary label by clinical dimentia rating (CDR)    
cdr = oasis_dataset.ext_vars['cdr'].astype(float)
cdr_numpy_arr = np.array(cdr)
for i in range(len(cdr_numpy_arr)):
    if(np.isnan(cdr_numpy_arr[i])): cdr_numpy_arr[i] = 1
    
    elif(cdr_numpy_arr[i] > 0.0): cdr_numpy_arr[i] = 1


# In[3]:


from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
rng = check_random_state(42)


# In[7]:


gm_imgs_train, gm_imgs_test, cdr_train, cdr_test = train_test_split(
    gm_imgs, cdr_numpy_arr, train_size=.7, random_state=rng)


# In[9]:


from nilearn import image
from nilearn import plotting 
# extract images at specific MNI coordinates and save in specified location         
i = 0
for image in gm_imgs_train:
    fig = plotting.plot_anat(image, display_mode = 'x', annotate= False, cut_coords = [40])
    if cdr_train[i] == 0:
        fig.savefig('C:\\Users\\user\\Neural Networks\\NiDataTrain\\not_impaired\\' + 'brain' + str(i) + '.png')
    elif cdr_train[i] == 1:
        fig.savefig('C:\\Users\\user\\Neural Networks\\NiDataTrain\\impaired\\' + 'brain' + str(i) + '.png')
    i = i + 1


# In[11]:


import os
os.getcwd()


# In[12]:


## Create location for test images to be saved 
try:
    os.makedirs("C:\\Users\\user\\Neural Networks\\NiDataTest\\not_impaired")
except OSError:
    print("error")
try:
    os.makedirs("C:\\Users\\user\\Neural Networks\\NiDataTest\\impaired")
except OSError:
    print("error")


# In[13]:


## extract test images from test sets 
      
i = 0
for image in gm_imgs_test:
    fig = plotting.plot_anat(image, display_mode = 'x', annotate= False, cut_coords = [40])
    if cdr_test[i] == 0:
        fig.savefig('C:\\Users\\user\\Neural Networks\\NiDataTest\\not_impaired\\' + 'brain' + str(i) + '.png')
    elif cdr_test[i] == 1:
        fig.savefig('C:\\Users\\user\\Neural Networks\\NiDataTest\\impaired\\' + 'brain' + str(i) + '.png')
    i = i + 1


# In[58]:


plot1 = plotting.plot_stat_map(img,
                       threshold=3, title="plot_stat_map",
                       cut_coords=[40, 8, 50])
plt.show()


# In[59]:


plot1 = plotting.plot_stat_map(img1,
                       threshold=3, title="plot_stat_map",
                       cut_coords=[40, 8, 50])
plt.show()

