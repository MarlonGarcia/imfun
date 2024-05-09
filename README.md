# Image Functions Library

This is a library to apply simple to complex functions to treat, transform and
prepare images. This library helps to prepare image data for machine learning,
for example, but also helps in simple functions, like image transformation,
loading, and printing. Next are an example of functions.


These are an example of simple functions:
- load_color_images: function to load all color images from a folder
- plot_color_images: function to print all color images from a folder
- highpass_fft: high-pass frequency filter using FFT algorithm
- highpass_gaus: high-pass frequency filter using Gaussian equation.
- flat2im: transform an image in flattened data, e.g. to enter machine-learning
algorithms (like Random Forest).
- beep: make a beep sequence to signalize.
- good_colormaps: show different colormaps to choose one to highlight features.


Examples of more complex functions:
- align_ECC: align images using ECC algorithm from OpenCV
- isoareas measure and statistics pixel's intensity by depth in a preferential
direction.
- im2label: transform an image to a segmented image label
- crop_multiple: crop multiple images with the same cropping area.
- polyroi: make a polygonal ROI in an image.
- crop_poly_multiple: make polygonal ROI and replicate the same ROI in other
images, changing its position.
- choose_points: choose points in an image, and retrieve its indexes.

OBS: some functions use libraries pynput and windsound, which some times are
difficult to install and do not works on non-windows platforms. Comment on
these imports if there are problems during installation.

## How to Install

You can install using `pip`:
> pip install image-functions

Author: Marlon Rodrigues Garcia  
Institution:  São Paulo State University  
Contact: marlonrg@gmail.com
