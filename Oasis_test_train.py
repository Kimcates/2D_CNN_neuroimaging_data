#!/usr/bin/env python
# coding: utf-8

"""
Oasis_test_train:
Loads NIFTI files and cdr + mmse info for each file in from nilearn.
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

#the directory that the NIFTI images will be saved to
data_dir = 'C:\\Nidata'

def NIFTI_to_PNG(path_list, dir):
    """
    :param  LOS  path_list: the list of paths to NIFTI images
    :param  string  dir: a specified path to create directory
    NIFTI_to_PNG: creates a directory to save NIFTI images as png files
                If the directory already exists, then that part is skipped.
                Each NIFITI image is augmented in 3 different ways and the 3
                resulting images are saved.  This is done to increase the
                size of the small sample of available NIFTI images
    """
    ## Create location for NIFTI images to be saved
    try:
        os.makedirs(dir)
    except OSError:
        print("error: directory already exists")

    #augment the images and save them as pngs
    count = 0 # a count of the files saved so far
    for path in gm_img_paths:
        #first augmentation -- color
        fig = plt.figure(figsize=(5, 7), facecolor='k')
        img_crop_1 = image.crop_img(path, rtol=1e-0001, copy=True)
        display = plotting.plot_img(img_crop_1, display_mode = 'z', cut_coords=[23], annotate = False, figure = fig)
        fig.savefig(dir + '\\' + 'brain' + str(count + 1) + '.png')
        count += 1
        plt.close(fig)

        # second augmentation -- added countours to color
        fig = plt.figure(figsize=(5, 7), facecolor='k')
        img_crop_2 = image.crop_img(path, rtol=1e-0001, copy=True)
        display_2 = plotting.plot_anat(img_crop_2, display_mode = 'z', cut_coords=[23], annotate = False, figure = fig)
        display_2.add_contours(img_crop_2, contours=1, antialiased=False,linewidths= 1., levels=[0], colors=['red'])
        display_2.add_contours(img_crop_2, contours=1, antialiased=False, linewidths=1., levels=[.3], colors=['blue'])
        display_2.add_contours(img_crop_2, contours=1, antialiased=False, linewidths=1., levels=[.5], colors=['limegreen'])
        fig.savefig(dir + '\\' + 'brain' + str(count + 1) + '.png')
        count += 1
        plt.close(fig)

        #third augmentation -- gray scale
        fig = plt.figure(figsize=(5, 7), facecolor='k')
        img_crop_3 = image.crop_img(path, rtol=1e-0001, copy=True)
        display_3 = plotting.plot_anat(img_crop_3, display_mode = 'z', cut_coords=[23], annotate = False, figure = fig)
        fig.savefig(dir + '\\' + 'brain' + str(count + 1) + '.png')
        count += 1
        plt.close(fig)

def load_images(dir):
    """
    return: list of nparrays: the list of nueral images
    load_images: loads the neural images from the given directory
    """
    #list of the neural images to be trained and tested in the model
    gm_imgs = []

    im_files_path = os.listdir(dir)
    IMG_SIZE = 64
    for image in im_files_path:
        img_array = cv2.imread(os.path.join(dir, image))
        img_array_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        gm_imgs.append(img_array_resized)

    return gm_imgs

def load_labels(df):
    """
    :param  DataFrame df: the dataframe containing the cdr and mmse data;
                            Labels are based on this data
    :return  list of int: list of target labels
    load_labels: iterates over given df and adds to list of targets;
                    because sample has been tripled, 3 of each label
                    are added at a time
    """
    #the list of labels for the nueral images
    target=[]

    for row, column in df.iterrows():
        if(column['cdr'] > 0.5 and column['mmse'] < 10):
            target.append(1)
            target.append(1)
            target.append(1)
        else:
            target.append(0)
            target.append(0)
            target.append(0)

    return target

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

#convert NIFTI objects to 3 png files each
NIFTI_to_PNG(gm_img_paths, data_dir)
#load the png files back in as a list of numpy arrays
gm_imgs = load_images(data_dir)
#load in labels for each image
labels = load_labels(df)
#reshape the images to fit the model
gm_imgs = np.array(gm_imgs).reshape(-1, 64, 64, 3)
#Divide the data into training and testing sets
rng = check_random_state(42)
gm_imgs_train, gm_imgs_test, labels_train, labels_test = train_test_split(
    gm_imgs, labels, train_size=.7, random_state=rng)

#save these 4 sets for reuse in model
pickle_save(gm_imgs_train, "gm_imgs_train")
pickle_save(gm_imgs_test, "gm_imgs_test")
pickle_save(labels_train, "labels_train")
pickle_save(labels_test, "labels_test")
