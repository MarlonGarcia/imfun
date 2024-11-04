'''Calculate pixels' statistics in detph (for any given direction)

This function loads all the images inside a folder, define a region of
interest inside each image (using 2 curves), devide this region in various
equally spaced areas (isoareas), and calculates statistics for the pixels'
intensity in each of these areas, following a particular direction (going
from the front curve defined to the back curve).

Applications:
-------------
    1) Calculate fluorescence inside a tumor, as a function of depth, on
    confocal microscopy or histological slides.
    
    2) Microscopy, medical imaging, material science, or any application
    that need to calculate pixels' statistics as a function of depth in any
    given direction.

Usage example:
--------------
    # Choose the folder where the images are:
    folder = r'C:/Users/user/data'
    
    # Choose the number of isoareas to calculate, for example 10:
    numb = 10
    
    # Call the function, defining the channels to enter in the statistics:
    dictionary = imfun.roi_stats_in_detph(folder, numb, channels=[1, 2, 3])

Detailed explanation:
---------------------
1. Loading: This function will enter the folder difined by the user in the
variable `folder`, as in the example above, and calculates the statistics
for each image inside it.

2. Choosing the Curves: When this function runs, a window will open with a
graphical user interface (GUI) that allow user to choose a "front curve"
and a "back curve" in the image. These two curves will define the area
where the statistics will be calculated (region of interest, ROI), which is
the area enclosed by these curves (area inside it). These curves ill also
determine the depth in which the statistics will be calculated, which will
be going from the "front curve" to the "back curve".

3. Isoareas: The closed region defined by the two curves drawn by the user
will be separated into various equally spaced areas called "isoarea", which
will follow a smooth transition between the curves.

4. The Mask: An additional mask will be choosen by the user. Only the
pixels inside this mask will be processed. Use this mask if you wants to
select just part of the region of interest defined by the two curves drawn
(to process just part of the isoareas). Otherwise, choose the intire region
of interest to process all the selected pixels (all isoareas). This mask is
important to eliminate "edge effects" of the statistics from the regions
where the curves touch (if needed).

5. Statistics in a Particular Direction: After that, a detailed statistics
will be calculated for each isoarea (mean, standard deviation, mode,
median, entropy), following a particular direction: going from the front
curve to the back curve. The number of isoareas is defined by `numb`.


Input Parameters
----------------
folder : string
    The directory where the images you want to process are. The images can
    be in different extensions, like LSM, PNG, JPG, etc.

numb : integer
    The number of isoareas you want to calculate and process.

Optional Parameters (kwargs)
----------------------------
channels : list
    List here all the channels to be processed, e.g. `channels = [1, 2, 3]`
    to process all the three channels of the image. Default value is `[1]`.
    If you want to process the first and the last channels of an image
    (which could be the `red` and the `blue` channel of an RGB image), you
    can call this function choosing `channels = [1, 3]`, for example. In
    the case of grayscale images, you can choose `channels = [1]`.

pixel_size : float or integer (default = 1.0)
    Enter the physical size discribed by a pixel. For example, if each
    pixel represents a size of 0.83 micrometers in the image (for a micro-
    scope image), than choose 'pixel_size = 0.83e-6'. Default value is 1.0.

show : boolean
    Choose 'True' to visualize each image processed, with its isoareas.

Returns (Outputs)
-----------------
dictionary : dictionary
    This function returns a dictionary with the statistics calculated for
    each channel selected in the variable 'channels'.'''


# Importing imfun library
import imfun

# Choosing the folder where the images are
folder = r'C:\Users\marlo\Downloads\Widefield\Confocal'

# Next we run the 'roi_stats_in_detph' function. This function is evaluated
# with "pixel_size = 0.83e-6", which means that the physical size measured by a
# pixel in the image has 0.83 micrometers (in Python, e-6 is 10^-6)
dictionary = imfun.roi_stats_in_detph(folder, 7, channels=[1], pixel_size=0.83e-6, show=True)



'''OBS: in this example the function is calculating all the statistics only for
the first channel of the image. If you want to calculate the statistics for all
the three channels, please choose `channels = [1, 2, 3]` in the code above'''