# -*- coding: utf-8 -*-
'''
Interactively labels sequence points on image profiles extracted from a
folder of grayscale images.

This function loads images from `image_dir`, allows the user to select
pixel intensity profiles from each image using a GUI, and then
interactively selects specific `numb_points` points on these profiles. The
labeled data is saved in a CSV file (`data.csv`), and the algorithm allows
the user for running this function repeatedly, and check which images were
already labeled before (by checking the `data.csv` file in the `save_dir`
folder).

Example of use:

    data = imfun.label_sequence_points(image_dir, 3, 3, save_dir=r'D:\Data\Sequences\OCT')

Parameters:
-----------
image_dir : str
    Path to the folder containing grayscale images to be labeled.
numb_points : int
    Number of points to select in each intensity profile.
numb_seq : int
    Number of intensity profiles to extract per image.
save_dir : str, optional (default=image_dir)
    Directory where the labeled data (`data.csv`) will be stored.
save_data : bool, optional (default=True)
    Whether to save the labeled data to a CSV file.
do_not_shuffle : bool, optional (default=False)
    If True, keeps image processing order instead of shuffling.

Workflow:
---------
1. Lists all image files in `image_dir`.
2. Removes already labeled images from the list (if `data.csv` exists).
3. Randomly shuffles images unless `do_not_shuffle` is set.
4. For each image:
   - Extracts `numb_seq` intensity profiles using `improfile` (user selects a profile line via GUI).
   - Displays the profile in an interactive plot where the user selects `numb_points` points.
   - Saves the image name, selected points, and full intensity profile to `data.csv`.

Output CSV Format:
------------------
The file `data.csv` is structured as follows:

| File name | Point 1 | Point 2 | ... | Point N | Profile (remaining columns) |
|----------|---------|---------|-----|---------|----------------------------|
| img1.png |  15     |  42     | ... |   78    | [pixel intensities]        |
| img2.png |  10     |  38     | ... |   72    | [pixel intensities]        |

Key Features:
-------------
- **Prevents re-labeling**: Images already labeled in `data.csv` are skipped.
- **Interactive Selection**: Users select both the profile and key points via a GUI.
- **Shuffling for fairness**: Randomly processes images unless disabled.
- **Flexible saving**: Users can disable CSV saving if needed.

Returns:
--------
data : list
    A list of labeled data, each entry containing:
    [image_name, selected_points..., intensity_profile]

'''

# Importing the ImFun library
import imfun


# First define the folder where images are (the `r` is used to prevent errors
# related with the use of slash or backslash) 
image_dir = r'D:\Data\Images\OCT\2025.02.22 - Luismar - Finger Print - Examples'

# Then define where to save the images
save_dir = r'D:\My Drive\College\Biophotonics Lab\Research\Programs\Python\Camera & Image\OCT\2025.02.22 - Find Epidermis and Dermis Width'


# Then run the function choosing 3 points in each sequence and 4 sequences for
# each image. The final data saved will containg the name of the image, the
# coordinate of 3 points, and 4 profiles chosen by each image:
data = imfun.label_sequence_points(image_dir, 3, 4, save_dir=save_dir)

