'''Function to create label images (or masks) from images in a folder.

This function creates another folder, with the same name as `root` plus the
string "labels", and saves the label images in this folder with the same
name of original images. Since labeling takes a lot of time, this function
can also identifies which images were alredy labeled before starting. The
final output image is scaled between 0 to 255, which can be changed by
setting `scale=False`. (TODO:enhance document., say about classes+1 or background)

Example:
--------
images = im2label(root, classes, show = True)

Parameters
----------
root : str
    Root directory where the images are located.

classes : int
    The number of classes to choose.

**kwargs : 
    Additional arguments to control various options:
    
    - save_images : bool (default: True)
        Choose if the label images will be saved in a folder. By default,
        the images are saved in a folder with the same name as `root` but
        adding 'labels' at the end of its name.
    
    - open_roi : str (default: None)
        If open_roi is not `None`, the algorithm chooses open regions (re-
        gions that end at the image boundary). If `open_roi` = 'above', the
        chosen region will be an open region above the selected area, the
        opposite happens if `open_roi` = 'below', with a region below the
        chosen points.
    
    - scale : bool (default: True)
        If True, the label images will be scaled between 0 and 256. The sa-
        ved labels/classes will be, e.g., 0, 127, and 255 in the case of 3
        classes. If scale=False, the images will be labeled with integer
        numbers, like 0, 1, and 2 in the case of 3 classes. In this case,
        the saved images might appear almost black in the folder.
    
    - label_names : list (default: None)
        A list of strings with the names of the labels to be shown during
        the interaction with the user.
    
    - cmap : int (cv2 colormap, default: None)
        Optional colormap to apply to the image.
    
    - show : bool (default: False)
        If True, shows the final image and its label until the user presses
        'ESC' or any key.
    
    - equalize : bool (default: False)
        If True, equalizes the grayscale image histogram.
    
    - color : tuple (default: (200, 200, 200))
        Enter a different color to color the working line (R, G, B) with
        values from 0 to 255.

Return
------
images : list
    A list with the labeled images, all of `numpy.ndarray` type.

Mouse actions:
- Left button: select a new point in the label;
- Right button: end selection and finish or go to another label;
- ESC: finish selection (close the algorithm).

Notes:
------
- When using `open_roi`, it is only possible to choose points from the left
  part of the image to the right.
- The remaining unlabeled pixels will be set as background pixels (they
  will belong to the last label). If a label is chosen more than once, the
  last chosen label will be applied.
- Images can be multidimensional (`[height, width, dimensions]`).
'''

import imfun2 as imfun



