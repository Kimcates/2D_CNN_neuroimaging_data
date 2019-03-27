#!/usr/bin/env python
# coding: utf-8

#Oasis_test_train:
"""
Loads NIFTI files and cdr info for each file in from nilearn.
Preprocesses this data for use in classification model.
Creates four files: "gm_imgs_train", "gm_imgs_test", "cdr_train", and "cdr_test"
"""
# Preliminaries
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from nilearn import datasets, image, plotting
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import pickle

# download oasis dataset
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects= 416)

gray_matter_map_filenames = oasis_dataset.gray_matter_maps
#a list of paths to the NIFTI images
gm_img_paths = gray_matter_map_filenames

#the neural images to be trained and tested on the model
gm_imgs = []

#the cdr for each image
#cdr: clinical dementia rating
#cdr is used to establish a label (0 or 1) for each image
#If a cdr is NAN or greater than 0, that nueral image
# is classified as positive for dimentia
cdr = oasis_dataset.ext_vars['cdr'].astype(float)
cdr_numpy_arr = np.array(cdr)

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
        print("error")
    #save the images as pngs

    count = 0 # a count of the files saved so far
    for path in gm_img_paths:
        fig = plotting.plot_anat(path, display_mode = 'x', annotate= False, cut_coords = [40])
        fig.savefig(dir + '\\' + 'brain' + str(count + 1) + '.png')
        count += 1
        fig.close()

def load_data(dir):
    """
    loads the neural images from the given dir
    Images are labled based on CDR
    gm_imgs and cdr_numpy_arr store the data for testing
    """
    im_files_path = os.listdir(dir)
    label_count = 0
    IMG_SIZE = 64
    for image in im_files_path:
        img_array = cv2.imread(os.path.join(data_dir, image))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        gm_imgs.append(img_array)

        if(np.isnan(cdr_numpy_arr[label_count])):
             cdr_numpy_arr[label_count] = 1
        elif(cdr_numpy_arr[label_count] > 0.0):
            cdr_numpy_arr[label_count] = 1

        label_count += 1


NIFTI_to_PNG(gm_img_paths, data_dir)
load_data(data_dir)

#Divide the data into training and testing sets
rng = check_random_state(42)
gm_imgs_train, gm_imgs_test, cdr_train, cdr_test = train_test_split(
    gm_imgs, cdr_numpy_arr, train_size=.7, random_state=rng)

#save these 4 sets for reuse in model
pickle_out = open("gm_imgs_train", "wb")
pickle.dump(gm_imgs_train, pickle_out)
pickle_out.close()

pickle_out = open("gm_imgs_test", "wb")
pickle.dump(gm_imgs_test, pickle_out)
pickle_out.close()

pickle_out = open("cdr_train", "wb")
pickle.dump(cdr_train, pickle_out)
pickle_out.close()

pickle_out = open("cdr_test", "wb")
pickle.dump(cdr_test, pickle_out)
pickle_out.close()
