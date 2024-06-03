# -*- coding: utf-8 -*-
''' imex_roi_stats.py

The following example show how to use the "roi_stats" function

Easely calculate statistics of images in a given region of the images

This function uses a interactive graphical user interface (GUI) to calcula-
te the statistics of multiple images, in a given region of interest (ROI)
inside the image. To use this function, your images have to be in a folder
tree like bellow. The outer folder will be 'Images Folder', that has to be
passed to this function in the 'images_dir' variable. Inside this folder
you can add as many parts of your experiment you want to (in this case
exemplified as animals). Inside each part (or animal) of your experiments,
it have to be folders corresponding to the exact experiments conducted (for
example different times, or differents treatments types or measurements).
After the processing, a '.csv' data file is saved in the folder/directory
specified in the variable 'save_dir'.

Images Folder
    |
    |
    |--- Animal 1
    |       |
    |       |--- Experiment 1
    |       |
    |       |--- Experiment 2
    |       |
    |       '--- Experiment 3
    |
    |
    '--- Animal 2
            |
            |--- Experiment 1
            |
            |--- Experiment 2
            |
            '--- Experiment 3

'''

import cv2
import imfun


# Enter the directory where the images are (e.g. CREME) in 'images_dir'
images_dir = r'C:\Users\marlo\Downloads\Widefield\Creme'

# Enter the directory to save the final data
save_dir = r'C:\Users\marlo\My Drive\College\Biophotonics Lab\Research\Programs\Python\Camera & Image\Campo Amplo\2023.08.24 - PDT Sara\results'

# Delineate here all the experiment names you want to
experiments = ['0h', '30min', '1h', '2h', 'pPDT']

# Delineate the colors you want to measure
colors = ['red', 'green', 'blue']

# Use the following if you want to analyse grayscale images
# colors = ['gray']

# Define the statistics you want to apply
stats = ['mean', 'std']

# Defining the colormap to be used to show the images
colormap = cv2.COLORMAP_PARULA

# Other good colormaps: "cv2.COLORMAP_PINK", "cv2.COLORMAP_HSV", "cv2.COLORMAP_BONE"

# Run this code to choose the folder by yourself
imfun.roi_stats(experiments, colors, colormap=colormap)

# Run this when you difined the directories in the variables 'images_dir' and 'save_dir'
# imfun.roi_stats(experiments, colors, save_dir=save_dir, images_dir=images_dir, colormap=colormap)

