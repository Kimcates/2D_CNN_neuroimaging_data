#!/usr/bin/env python
# coding: utf-8

"""
Oasis_test_train:
Loads NIFTI files and cdr info for each file in from nilearn.
Preprocesses this data for use in classification model.
Creates four files using pickle: "gm_imgs_train", "gm_imgs_test", "cdr_train", and "cdr_test"
"""
# Preliminaries
import numpy as np
import os
import cv2
from nilearn import datasets, image, plotting
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# download oasis dataset
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects= 416)

gray_matter_map_filenames = oasis_dataset.gray_matter_maps
#a list of paths to the NIFTI images
gm_img_paths = gray_matter_map_filenames

#the neural images to be trained and tested on the model
gm_imgs = []

#the cdr and mmse for each image
#cdr: clinical dementia rating
#mmse: Mini-Mental State Exam
#if subject's cdr is greater than 0.5 and mmse is
# greater than 10, subject is positive for dimentia
cdr = oasis_dataset.ext_vars['cdr'].astype(float)
mmse = oasis_dataset.ext_vars['mmse'].astype(float)
cdr_numpy_arr = np.array(cdr)
mmse_numpy_arr = np.array(mmse)

data = list(zip(cdr_numpy_arr, mmse_numpy_arr))

df = pd.DataFrame(data, columns = ['cdr', 'mmse'])
#all nan values can be replaced with 1 for the purpose of this model
df = df.fillna(1)
#target is the list of labels for the nueral images
target=[]

#the directory that the NIFTI images will be saved to
data_dir = 'C:\\Nidata'

def NIFTI_to_PNG(path_list, dir):
    """
    :param  LOS  path_list: the list of paths to NIFTI images
    :param  string  dir: a specified path to create directory
    NIFTI_to_PNG: creates a directory to save NIFTI images as png files
                If the directory already exists, then that part is skipped.
    """
    ## Create location for NIFTI images to be saved
    try:
        os.makedirs(dir)
    except OSError:
        print("error: directory already exists")

    #augment the images and save them as pngs
    count = 0 # a count of the files saved so far
    for path in gm_img_paths:
        fig = plt.figure(figsize=(5, 7), facecolor='k')
        img_crop = image.crop_img(img, rtol=1e-0001, copy=True)
        display = plotting.plot_anat(img_crop, display_mode = 'z', cut_coords=[23], annotate = False, figure = fig)
        display.add_contours(img_crop, contours=1, antialiased=False,linewidths= 1., levels=[0], colors=['red'])
        display.add_contours(img_crop, contours=1, antialiased=False, linewidths=1., levels=[.3], colors=['blue'])
        display.add_contours(img_crop, contours=1, antialiased=False, linewidths=1., levels=[.5], colors=['limegreen'])
        fig.savefig(dir + '\\' + 'brain' + str(count + 1) + '.png')
        count += 1
        fig.close()

def load_data(dir):
    """
    loads the neural images from the given directory
    Images are labled based on CDR
    gm_imgs and cdr_numpy_arr store the data for testing
    """
    im_files_path = os.listdir(dir)
    IMG_SIZE = 64
    for image in im_files_path:
        img_array = cv2.imread(os.path.join(dir, image))
        img_array_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        gm_imgs.append(img_array_resized)

        for row, column in df.iterrows():
            if(column['cdr'] > 0.5 and column['mmse'] < 10):
                target.append(1)
            else:
                target.append(0)

def pickle_save(pythonOBJ, fileName):
    """
    :param  pythonOBJ:    Any python object
    :param  string  file: the name of the file to save the object to
    pickle_save:    Uses Pickle to save a python obj to a specified file
    """
    try:
        pickle_out = open(fileName, "wb")
        pickle.dump(pythonOBJ, pickle_out)
        pickle_out.close()
    except:
        print("error: could not open " + fileName)

#convert NIFTI images to png files
NIFTI_to_PNG(gm_img_paths, data_dir)
#load the png files back in as numpy arrays and
# fill gm_imgs with these arrays
#Also fill cdr_numpy_arr with the label for each image
load_data(data_dir)
#reshape the images to fit the model
gm_imgs = np.array(gm_imgs).reshape(-1, 64, 64, 3)
#Divide the data into training and testing sets
rng = check_random_state(42)
gm_imgs_train, gm_imgs_test, cdr_train, cdr_test = train_test_split(
    gm_imgs, cdr_numpy_arr, train_size=.7, random_state=rng)

#save these 4 sets for reuse in model
pickle_save(gm_imgs_train, "gm_imgs_train")
pickle_save(gm_imgs_test, "gm_imgs_test")
pickle_save(cdr_train, "cdr_train")
pickle_save(cdr_test, "cdr_test")
