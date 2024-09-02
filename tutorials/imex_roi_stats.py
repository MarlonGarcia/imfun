''' Easily calculate statistics of images in a given region of the images

This function uses an interactive graphical user interface (GUI) to calcula-
te the statistics of multiple images, in a given region of interest (ROI)
inside the image. To use this function, your images must be in a folder
tree like below. The outer folder will be 'Images Folder', which has to be
passed to this function in the 'images_dir' variable. Inside this folder,
you can add as many parts of your experiment as you want to (in this case
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

Parameters
----------
experiments : list
    Names of the experiments. Note that it has to mach the experiment names
    e.g. ['Experiment 1', 'Experiment 2', 'Experiment 3'] in the example.
colors : list
    Name of the colors (or channels) to be analized in the image, or a list
    with the string 'gray' for grayscale images. E.g.: ['gray'] or ['red'],
    or even all the colors ['red', 'green', 'blue']

**kwargs (arguments that may or may not be passed to the function)
----------
images_dir : string
    Root directory (outer folder) of your images. E.g. 'C:/Users/User/data'
save_dir : string
    Directory to save the images.
stats : list
    A list with the statistics to be calculated. Only mean and standard
    deviation are suported until now. E.g. ['mean'] or ['mean', 'std']
show : boolean
    If 'True', print all the images processed. The default is 'False'.
colormap : int
    The colormap to use while choosing the region of interest
    examples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV, cv2.COLORMAP_PARULA.'''

import cv2
import imfun


# Enter the directory where the images are (e.g. CREME) in 'images_dir'
images_dir = r'C:\Users\marlo\Downloads\Widefield\Creme'

# Enter the directory to save the final data
save_dir = r'C:\Users\marlo\Downloads\Widefield\Results'

# Delineate here all the experiment names you want to
experiments = ['0h', '30min', '1h', '2h', 'pPDT']

# Delineate the colors you want to measure
colors = [1, 2, 3]
         
# To analyze grayscale images, choose any number from 1 to 3 as follows
# colors = [1]

# Define the statistics you want to apply. In this case we applied all of them
statistics = ['mean', 'std', 'median', 'mode', 'entropy']

# Defining the colormap to be used to show the images
colormap = cv2.COLORMAP_PARULA

# Other colormaps: "cv2.COLORMAP_PINK", "cv2.COLORMAP_HSV", "cv2.COLORMAP_BONE"

# Run the code below to choose the folder by yourself (with a pop-up)
# imfun.roi_stats(experiments, colors, stats=stats, colormap=colormap)

# Run the code below when you want to difine the directories in the variables
# 'images_dir' and 'save_dir'. Using 'show=True', all images will be printed.
imfun.roi_stats(experiments, colors, statistics=statistics, save_dir=save_dir, images_dir=images_dir, show=True, colormap=colormap)
