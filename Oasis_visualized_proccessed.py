#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


# download oasis dataset on working directory 
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects= 100)


# In[7]:


from nilearn import image
from nilearn import plotting 
# index first image in oasis dataset - get path with gray_matter_maps method 
img = oasis_dataset.gray_matter_maps[0]
## MNI coordinates for AD specified by cut_coords = 40, 8, 50 
plotting.plot_anat(img, title="plot_anat", cut_coords=[40, 8, 50])
plt.show()


# In[8]:


img


# In[9]:


# image smoothed
img_smoothed =  image.smooth_img(img, fwhm=5)
plotting.plot_anat(img_smoothed, title="smoothed - CDR = 0", cut_coords=[40, 8, 50])
plt.show()


# In[64]:


img_smoothed1 =  image.smooth_img(img1, fwhm=5)
plotting.plot_anat(img_smoothed, title="smoothed - CDR = 1", cut_coords=[40, 8, 50])
plt.show()


# In[29]:


# not smoothed 
plot1 = plotting.plot_stat_map(img,
                       threshold=3, title="plot_stat_map",
                       cut_coords=[40, 8, 50])
plt.show()


# In[30]:


# smoothed 
plot2 = plotting.plot_stat_map(img_smoothed,
                       threshold=3, title="plot_stat_map",
                       cut_coords=[40, 8, 50])
plt.show()


# In[27]:


## z - axial 
plotting.plot_img(img, display_mode = 'z')
plt.show()


# In[66]:


plotting.plot_img(img1, display_mode = 'z')
plt.show()


# In[28]:


## y - coronal 
plotting.plot_img(img, display_mode = 'y')
plt.show()


# In[65]:


img1 = img = oasis_dataset.gray_matter_maps[24]
plotting.plot_img(img1, display_mode = 'y')
plt.show()


# In[67]:


## x - sagittal 
plotting.plot_img(img, display_mode = 'x')
plt.show()


# In[68]:



plotting.plot_img(img1, display_mode = 'x')
plt.show()


# In[54]:


img1 = oasis_dataset.gray_matter_maps[24]
## MNI coordinates for AD specified by cut_coords = 40, 8, 50 
plotting.plot_anat(img1, title="CDR = 1 ", cut_coords=[40, 8, 50])
plt.show()


# In[57]:



## MNI coordinates for AD specified by cut_coords = 40, 8, 50 
plotting.plot_anat(img, title="CDR = 0", cut_coords=[40, 8, 50])
plt.show()


# In[74]:


## plot image for one direction at specific coords 
plotting.plot_anat(img, title="CDR = 0", display_mode = 'x', cut_coords=[40])
plt.show()


# In[75]:


plotting.plot_anat(img1, title="CDR = 1", display_mode = 'x', cut_coords=[40])
plt.show()


# In[79]:


fig = plotting.plot_anat(img1, title="CDR = 1", display_mode = 'x', cut_coords=[40])
fig.savefig('brain1.png')
plt.show()


# In[2]:


import os
os.getcwd()


# In[10]:


## Create location for images to be saved 
try:
    os.makedirs("C:\\Users\\kimbe\\NN\\NiData\\not_impaired")
except OSError:
    print("error")
try:
    os.makedirs("C:\\Users\\kimbe\\NN\\NiData\\impaired")
except OSError:
    print("error")


# In[11]:


# create binary label by clinical dimentia rating (CDR)    
cdr = oasis_dataset.ext_vars['cdr'].astype(float)
cdr_numpy_arr = np.array(cdr)
for i in range(len(cdr_numpy_arr)):
    if(np.isnan(cdr_numpy_arr[i])): cdr_numpy_arr[i] = 1
    
    elif(cdr_numpy_arr[i] > 0.0): cdr_numpy_arr[i] = 1


# In[ ]:


# extract images at specific MNI coordinates and save in specified location         
i = 0
for image in oasis_dataset.gray_matter_maps:
    fig = plotting.plot_anat(image, display_mode = 'x', cut_coords = [40])
    if cdr_numpy_arr[i] == 0:
        fig.savefig('C:\\Users\\kimbe\\NN\\NiData\\not_impaired\\' + 'brain' + str(i) + '.png')
    elif cdr_numpy_arr[i] == 1:
        fig.savefig('C:\\Users\\kimbe\\NN\\NiData\\impaired\\' + 'brain' + str(i) + '.png')
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

