# Image Functions Library

Library for image pre-processing, with functions to handle and prepare images
for machine learning and image processing. This library accounts for functions
to: load and plot a group of images, pre-processing, choose ROI regions (may be
polygonal), choose points, get image properties, align and transform images
(including rotate, scale, etc.), filter signals and images (2D data), among
others.


### 1. Functins to Load and Plot
    - load_gray_images: loads all images from a folder, in grayscale;
    - load_color_images: loads all color images from a folder;
    - plot_gray_images: prints all grayscale images from a variable 'I';
    - plot_color_images: prints all color images from a variable 'I';
    - plot_gray: prints a grayscale image;
    - plot_bgr: prints a color image in BGR format;


### 2. Pre-Processing for Machine Learning and Computer Vision

#### 2.1. ROI and Handling (*Most Important Ones*)
    - polyroi: GUI to creates a polygonal region of interest (ROI)
    - crop_image: GUI to creates a rectangular crop in an image
    - crop_multiple: crops multiple images using the same crop from 1st image
    - crop_poly_multiple: polygonal crop multiple images based on 1st cropping
    - choose_points: GUI to interact with the user to choose points in an image
    - imchoose: function to chose images in a given set of images (with GUI)
    - imroiprop: getting properties from an image ROI
    
    
#### 2.2. Image Alignment and Transformation
    - rotate2D: rotate points by an angle about a center;
    - flat2im: transforms a flat vector into a 2D image
    - im2flat: transforms a 2D image in a flat vector
    - im2label: GUI to transforms images in labels for image segmentation (*folder, new folder)
    - scale255: scales an image to the [0, 255] range
    - align_features: Aligh images with Feature-Based algorithm, from OpenCV (maybe not working)
    - align_ECC: image alignment using ECC algorithm from OpenCV (difuse image)
    - imwarp: function to warp a set of images using a warp matrix (maybe not working)
    
    
### 3. Filtering Images and Signals
    - filter_finder: study and find which filter to use (for signals, 1D)
    - highpass_gauss: high-pass Gaussian filter for images (2D)
    - highpass_fft: high-pass image (2D) filter based on FFT
    - lowpass_fft: low-pass image (2D) filter based on FFT
    - filt_hist: filtering histograms with zero/null values (removing zeros)


### 4. Bonus Functions
    - beep: making 'beeps' to help warning when a long algorithm has finished;
    - isoareas: complex function to measure pixels' intensity in adjacet areas ***
    - good_colormaps: visualizing best Matplotlib colormaps in an image
    - improfile: finds the pixels' intensity profile between two points (GUI) (maybe not working)
    

## How to Install

You can install using `pip`:

```
pip install image-functions
```

*OBS*: some functions use the 'pynput' and 'windsound' libraries, which may be
difficult to install and do not works on non-windows platforms. Comment these
library imports if there are problems during installation or loading.

- author: Marlon Rodrigues Garcia
- contact: marlon.garcia@unesp.br
- institution: Sao Paulo State University (Unesp)


### Scientific Research

This work is the product of the research being conducted at two universities at Brazil:

- Dept. of Electronic and Telecommunication Engineering
- School of Engineering, Campus of Sao Joao da Boa Vista
- Sao Paulo State University (Unesp)
- website: https://www.sjbv.unesp.br/

- Biophotonics Laboratory, Optics Group (GO)
- São Carlos Institute of Physics (IFSC)
- University of São Paulo (USP)
- website: https://www2.ifsc.usp.br/english/