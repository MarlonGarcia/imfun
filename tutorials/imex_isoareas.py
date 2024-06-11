'''This function loads all the images inside a folder, define a region of
interest inside each image (using two lines), devide this region in various
equally spaced areas (isoareas), and calculates statistics for the pixels'
intensity in each of these area, following a particular direction (going
from one of the lines defined to the other one).

Applications:
    Calculate fluorescence inside a tumor, as a function of depth, in his-
    tological slides.
    Microscopy, medical imaging, or material science, application that cal-
    culates pixels' statistics for different position in a given direction.

Usage example:
    # Choose the folder where the images are
    folder = r'C:/Users/user/data'
    
    # Choose the number of isoareas to calculate
    numb = 10
    
    # Call the function, defining the channels to enter in the statistics
    dictionary = roi_stats_in_detph(folder, numb, channels=[1, 2, 3])

Detailed explanation:
    
1. Loading: This function will enter the folder difined in the variable
'folder', as in the above example, and process all the images inside it.

2. Choosing Lines: Then you will choose two lines (the front line and the
back one), using a graphical user interface (GUI). These lines will define
the region where the statistics will be calculated.

3. Isoareas: The closed region defined by the two lines drawn by the user
will be separated into various equally spaced lines (isolines). The area
defined between two adjascent 'isolines' will be called an 'isoarea'.

4. Statistics in a Particular Direction: After that, a detailed statistics
will be calculated for each isoarea (mean, standard deviation, mode, median
and entropy), following a particular direction: going from the front line
to the back line. The number of isoareas is defined in 'numb'.


Input Parameters
----------------
folder : string
    The directory where the images you want to process are.

numb : integer
    The number of isoareas you want to calculate and process.

Optional Parameters (kwargs)
----------------------------
channels : list
    List here all the channels to be processed, e.g. 'channels = [1, 2, 3]'
    to process all the three channels of an image. Default value is '[1]'.
    In the case of grayscale images, you can use 'channels = [1]'.

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

# Running the 'roi_stats_in_detph' function
dictionary = imfun.roi_stats_in_detph(folder, 7, channels=[1], pixel_size = 0.83e-6, show = True)



'''OBS: the function is calculating the statistics only for the first channel
of the image. If you want to calculate the statistics for all the three
channels, choose 'channels = [1, 2, 3]' in the code above'''