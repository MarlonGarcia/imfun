# -*- coding: utf-8 -*-
"""
Imaging Functions Library

This is a Python library to apply simple to complex functions to treat, trans-
form and prepare images. This library helps to prepare imaging data, for
example, for machine learning, but also helps in simple functions, like image
transformation, loading, and printing. Next are the example of some functions.


Next are example of some simple functions from this library:
- load_color_images: function to load all color images from a folder
- plot_color_images: function to print all color images from a folder
- highpass_fft: high-pass frequency filter using FFT algorithm
- highpass_gaus: high-pass frequency filter using Gaussian equation.
- flat2im: transform an image in flattened data, e.g. to enter machine-learning
algorithms (like Random Forest).
- beep: make a beep sequence to signalize.
- good_colormaps: show different colormaps to choose one to highlight features.


Examples of some more complex functions this library can perform:
- align_ECC: align images using ECC algorithm from OpenCV
- isoareas: measure and statistics pixel's intensity by depth from a tumor
(e.g. for a tumor slide) in a preferential directon (for example for fluore
scence images)
- im2label: transform an image to a segmented image label
- crop_multiple: crop multiple images with the same cropping area.
- polyroi: make a polygonal ROI in an image.
- crop_poly_multiple: make polygonal ROI and replicate the same ROI in other
images, changing its position.
- choose_points: choose points in an image, and retrieve its indexes.

OBS: some functions use libraries pynput and windsound, which some times are
difficult to install and do not works on non-windows platforms. Comment on
these imports if there are problems during installation.

@author: Marlon Rodrigues Garcia
@instit.:  State University of São Paulo
@contact: marlonrg@gmail.com"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats
# To use a 'beep' sound, uncomment: winsound, time.
import winsound               # it only works on windows
import time
import ctypes
from pynput import keyboard   # It does not worked on colabs
from random import shuffle
from scipy import fftpack     # to apply FFT and iFFT
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d


def load_gray_images(folder,colormap):
    """Loading grayscale images from 'folder'
    
    This function load all the images from 'folder' in grayscale and store in
    variable 'I'.
    
    if colormap = -1, no colormap is assigned
    if colormap = cv2_COLORMAP_OPTION, (being option = hsv, jet, parula, etc),
    or a colormap reference number (0, 1, 2, etc), the colormap chosen option
    is assigned.
    """
    I = []
    flag1 = cv2.IMREAD_GRAYSCALE
    if colormap == -1 or colormap == None:
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename), flag1)
            if img is not None:
                I.append(img)
    else:
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename), flag1)
            if img is not None:
                img2 = cv2.applyColorMap(img, colormap)
                I.append(img2)
    return I



def load_color_images(folder):
    """Loading colorful images from 'folder'
    
    This function load all colorful images from 'folder' in variable I.
    """
    I = []
    flag1 = cv2.IMREAD_COLOR
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), flag1)
        if img is not None:
            I.append(img)
    return I



def plot_gray_images(I,n):
    '''Program to plot 'n' images from 'I' using 'opencv2'
    
    This program will plot 'n' images from variable the list 'I' (a list of 
    numpy arrays). Press 'ESC' for close all the windows, or another key to 
    mantain the windows.
    '''
    I1 = np.asarray(I)
    for count in range(0,n):
        # We use cv2.WINDOW_AUTOSIZE to allow changes in image size.
        # If we use  WINDOW_NORMAL, we cannot change the image size (maximize).
        flag3 = cv2.WINDOW_NORMAL
        name = 'Figure ' + str(count+1)
        cv2.namedWindow(name, flag3)
        cv2.imshow(name, I1[count])
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # Waiting for 'ESC' key before close all the windows
        cv2.destroyAllWindows()



def plot_color_images(I,n):
    '''Program to plot 'n' color images from 'I' using 'opencv2'
    
    This program will plot 'n' color images from variable the list 'I' (a list
    of numpy arrays). Press 'ESC' for close all the windows, or another key to
    mantain the windows.
    '''
    I1 = np.asarray(I)
    for count in range(0,n):
        # We use cv2.WINDOW_AUTOSIZE to allow changes in image size.
        # If we use  WINDOW_NORMAL, we cannot change the image size (maximize).
        flag3 = cv2.WINDOW_NORMAL
        name = 'Figure ' + str(count+1)
        cv2.namedWindow(name, flag3)
        cv2.imshow(name, I1[count])
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # Waiting for 'ESC' key before close all the windows
        cv2.destroyAllWindows()



def plot_gray(I, name, colormap):
    '''Program to plot gray images with 'matplotlib' from the list 'I'
    
    I: input image (as a 'list' variable)
    name: window name
    colormap: a colormap name (pink, RdGy, gist_stern, flag, viridis, CMRmap)
    
    This program will plot gray images from the list in 'I', using matplotlib.
    '''
    if name is None:
        name = 'Figure'
    I1 = np.asarray(I)
    shape = I1.shape
    if len(shape) == 2:
        plt.figure(name)
        plt.imshow(I1, colormap)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
    else:
        print('\nYour variable "I" is not a recognized image\n')



def plot_bgr(I,name):
    '''Program to plot BGR images with 'matplotlib' from the list 'I'
    
    This program will plot RGB images from the list in 'I', using matplotlib.
    '''
    if name is None:
        name = 'Figure'
    I1 = np.asarray(I)
    shape = I1.shape
    if len(shape) == 3:
        plt.figure(name)
        plt.imshow(I1[:,:,::-1])
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
    else:
        print('\nYour variable "I" is not a recognized image\n')



def beep(**kwargs):
    ''' Function to make a beep
    beep(freq,duration)
    
    **freq: tone frequency, in hertz (preseted to 2500 Hz)
    **duration: tone duration, in miliseconds (preseted to 300 ms)
    '''
    freq = kwargs.get('freq')
    duration = kwargs.get('duration')
    if freq is None:
        freq = 2500  # Set Frequency To 2500 Hertz
    if duration is None:
        duration = 300  # Set Duration To 300 ms == 0.3 second
    numb = 5 # number of beeps
    for n in range(0,numb):
        time.sleep(0.0005)
        winsound.Beep(freq, duration)



def rotate2D(pts, cnt, ang):
    '''Rotating the points about a center 'cnt' by an ang 'ang' in radians.
    
    [pts_r] = rotate2D(pts, cnt, ang)
    
    '''
    return np.dot(pts-cnt,np.array([[ np.cos(ang),np.sin(ang)],
                                    [-np.sin(ang),np.cos(ang)]]))+cnt



class choose_points1(object):
    '''This is the class to help 'choose_points' function.
    '''
    def __init__(self):
        self.done = False       # True when we finish the polygon
        self.current = (0, 0)   # Current position of mouse
        self.points = []        # The chosen vertex points
    
    # This function defines what happens when a mouse event take place.
    def mouse_actions(self, event, x, y, buttons, parameters):
        # If polygon has already done, return from this function
        if self.done:
            return
        # Update the mouse current position (to draw a 'plus symbol' in image).
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate a new point added to 'points' .
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        # Right buttom pressed indicate the end of the choose, so 'done = True'
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True
    
    # This function is to really 'run' the 'choose_points1' class:
    def run(self, image, cmap, window_name, img_type, show):
        # If there is no a window name chose, apply the standard one.
        if window_name is None:
            window_name = "Choose points"
        # Separating if we use or not a colormap.
        if cmap is not None:
            image_c = cv2.applyColorMap(image, cmap)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        # function to make the window with name 'window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(window_name, self.mouse_actions)
        
        # loop to draw the lines while we are choosing the polygon vertices
        while(not self.done):
            # make a copy to draw the working line
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            
            # Defining thickness based on image size (to lines in image)
            if np.shape(image2)[0] > 400:
                radius = int(np.shape(image2)[1]/200)
            else:
                radius = 2
            
            # If at least 1 point was chosen, draw the points. We use
            # cv2.imshow to constantly show a new image with 
            # the chosen points.
            ### Next 'circle' is disabled, to does not show the current point:
            # image2 = cv2.circle(image2, self.current, radius=radius,
            #                     color=(222, 222, 252), thickness=radius-2)
            if (len(self.points) > 0):
                for n in range(0,len(self.points)):
                    cv2.circle(image2, self.points[n], radius=radius,
                               color=(232, 222, 222), thickness=radius-2)
                cv2.imshow(window_name, image2)
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                self.done = True
        
        # When leaving loop, draw the polygon IF at least a point was chosen
        if (len(self.points) > 0):
            for n in range(0,len(self.points)):
                cv2.circle(image2, self.points[n], radius=radius,
                           color=(232, 222, 222), thickness=radius-2)
            cv2.imshow(window_name, image2)
        
        return image2, np.asarray(self.points), window_name    



def choose_points(image, **kwargs):
    '''This function return the local of chosen points.
    [image_out, points] = choose_points(image, **cmap, **window_name, **show)
    
    cmap: Chose the prefered colormap. If 'None', image in grayscale.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    window_name: choose a window name if you want to
    show: if True, it shows the image with the points remains printed.
    image_out: the output polygonal ROI
    points: the chosen vertices (numpy-array)
    
    left buttom: select a new vertex
    right buttom: finish the vertex selection
    ESC: finish the function
    
    With this function it is possible to choose points in an image, and
    to get their positions.
    
    **The files with double asteristic are optional (**kwargs).    '''   
    choose_class = choose_points1()
    
    # With 'kwargs' we can define extra arguments that the user can input.
    cmap = kwargs.get('cmap')
    window_name = kwargs.get('window_name')
    show = kwargs.get('show')
    
    # Discovering the image type [color (img_type1=3) or gray (img_type1=2)]
    img_type1 = len(np.shape(image))
    if img_type1 == 2:
        img_type = 'gray'
    else:
        img_type = 'color'
    
    # 'policlass.run' actually run the program we need.
    [image2, points, window_name] = choose_class.run(image, cmap,
                                                   window_name, img_type, show)
    # If show = True, mantain the image ploted.
    if show != True:
        cv2.waitKey(500)    # Wait a little to user see the chosen points.
        cv2.destroyWindow(window_name)
    
    return image2, points



def flat2im(flat, height, width):
    ''' Convert 1D flat 'numpy-array' into an 2D image
    I = flat2im(flat, height, width)
    
    flat: 1D array with size 'height*width'
    I: output 2D image with shape 'height' by 'width'
    '''
    
    # if flat have multichennels, we have to account for (e.g. RGB)
    try: channels = np.shape(flat)[1]
    except: channels = None
    
    if (channels == 1) or (channels == None):
        I = np.zeros((height, width), np.uint8)
        if channels == 1:
            for l in range(0, height):
                I[l,:] = flat[l*width:(l+1)*width, 0]
        else:
            for l in range(0, height):
                I[l,:] = flat[l*width:(l+1)*width]
    else:
        I = np.zeros((height, width, channels), np.uint8)
        for c in range(0, channels):
            for l in range(0, height):
                I[l,:,c] = flat[l*width:(l+1)*width, c]
        
    return I



def im2flat(image):
    '''
    Parameters
    ----------
    image : np.array
        A 2D or 3D numpy array (if not, the program try to convert)
        This is a numpy image to be flatten.

    Returns
    -------
    flat : np.array
        A flatten array with (high*width)x1 for a 2D input, and a (high*width)xN for a N-dimensional input.
    '''
    image = np.asarray(image)
    
    if len(np.shape(image)) == 2:
        # first we define the first part of flat (the first line of 'image')
        flat = np.array(image[0,:])
        # since we already saved the first line of 'image', we iterate 
        # until all the images minus one, as follows.
        for l in range(len(image[:,0])-1):
            flat = np.concatenate((flat,image[l+1,:]))
    elif len(np.shape(image)) > 2:
        flat = []
        for d in range(len(image[0,0,:])):
            flat_temp = np.array(image[0,:,d])
            for l in range(len(image[:,0,d])-1):
                flat_temp = np.hstack((flat_temp, image[l+1,:,d]))
            flat.append(flat_temp)
        flat = np.asarray(flat)
    else: print('The input \'image\' has to be at least 2D')
        
    return flat



class im2label_class(object):
    '''This is the class helps 'im2label' function.
    '''
    def __init__(self):
        self.done = False       # True when we finish the polygon
        self.current = (0, 0)   # Current position of mouse
        self.points = []        # The chosen vertex points
    
    # This function defines what happens when a mouse event take place.
    def mouse_actions(self, event, x, y, buttons, parameters):
        # If polygon has already done, return from this function
        if self.done:
            return
        # Update the mouse current position (to draw a 'plus symbol' in image).
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate a new point added to 'points' .
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        # Right buttom pressed indicate the end of the choose, so 'done = True'
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True
    
    # This function is to really 'run' the 'choose_points1' class:
    def run(self, image, label, cmap, w, h, dim, img_type, color):
        # Stating a window_name for opencv
        window_name = 'Choose a region for label ' + str(label)
        
        # Separating if we use or not a colormap.
        if cmap is not None:
            image_c = cv2.applyColorMap(image, cmap)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        # function to make the window with name 'window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(window_name, self.mouse_actions)
        
        # Defining thickness based on image size (to lines in image)
        if w > 500:
            thickness = int(w/500)
        else:
            thickness = 1
        
        # loop to draw the lines while we are choosing the polygon vertices
        while(not self.done):
            # make a copy to draw the working line
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            # Creating the zoom in figure, we used 2*int because int(w/3) can
            # be different from 2*int(w/6).
            zoom_in = np.zeros([2*int(h/6), 2*int(w/6), dim], np.uint8)
            
            # If at least 1 point was chosen, draw the polygon and the working
            # line. We use cv2.imshow to constantly show a new image with the
            # vertices already drawn and the updated working line
            if (len(self.points) > 0):
                # Writing lines in big figure
                cv2.polylines(image2, np.array([self.points]), False, color,
                              thickness)
                cv2.line(image2, self.points[-1], self.current, color,
                         thickness)
                if self.current[1]-int(h/6) < 0:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[int(h/6)-self.current[1]:,int(w/6)-self.current[0]:,:] = image2[0:self.current[1]+int(h/6),0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[int(h/6)-self.current[1]:,0:int(w/6)+w-self.current[0],:] = image2[0:self.current[1]+int(h/6),self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[int(h/6)-self.current[1]:,:,:] = image2[0:self.current[1]+int(h/6),self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                elif self.current[1]+int(h/6) > h:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[0:int(h/6)+h-self.current[1],int(w/6)-self.current[0]:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[0:int(h/6)+h-self.current[1],0:int(w/6)+w-self.current[0],:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[0:int(h/6)+h-self.current[1],:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                else:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[:,int(w/6)-self.current[0]:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[:,0:int(w/6)+w-self.current[0],:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[:,:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                
                # Making a 'cross' signal to help choosing region
                zoom_in[int(h/6),int(4*w/30):int(19*w/120)] = np.uint8(zoom_in[int(h/6),int(4*w/30):int(19*w/120)]+125)
                zoom_in[int(h/6),int(21*w/120):int(6*w/30)] = np.uint8(zoom_in[int(h/6),int(21*w/120):int(6*w/30)]+125)
                zoom_in[int(4*h/30):int(19*h/120),int(w/6)] = np.uint8(zoom_in[int(4*h/30):int(19*h/120),int(w/6)]+125)
                zoom_in[int(21*h/120):int(6*h/30),int(w/6)] = np.uint8(zoom_in[int(21*h/120):int(6*h/30),int(w/6)]+125)

                # Transforming 'zoom_in' is a zoom (it is a crop right now)
                h_z, w_z = np.shape(zoom_in)[0],  np.shape(zoom_in)[1]
                zoom_in = cv2.resize(zoom_in[int(h_z/2)-int(h_z/4):int(h_z)-int(h_z/4), int(w_z/2)-int(w_z/4):int(w_z)-int(w_z/4)], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                image2[0:2*int(h/6),w-2*int(w/6):w]=zoom_in
                cv2.imshow(window_name, image2)
            else:
                if self.current[1]-int(h/6) < 0:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[int(h/6)-self.current[1]:,int(w/6)-self.current[0]:,:] = image2[0:self.current[1]+int(h/6),0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[int(h/6)-self.current[1]:,0:int(w/6)+w-self.current[0],:] = image2[0:self.current[1]+int(h/6),self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[int(h/6)-self.current[1]:,:,:] = image2[0:self.current[1]+int(h/6),self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                elif self.current[1]+int(h/6) > h:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[0:int(h/6)+h-self.current[1],int(w/6)-self.current[0]:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[0:int(h/6)+h-self.current[1],0:int(w/6)+w-self.current[0],:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[0:int(h/6)+h-self.current[1],:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6)+int(h/6)-h+self.current[1],self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                else:
                    if self.current[0]-int(w/6) < 0:
                        zoom_in[:,int(w/6)-self.current[0]:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),0:self.current[0]+int(w/6),:]
                    elif self.current[0]+int(w/6) > w:
                        zoom_in[:,0:int(w/6)+w-self.current[0],:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),self.current[0]-int(w/6):w,:]
                    else:
                        zoom_in[:,:,:] = image2[self.current[1]-int(h/6):self.current[1]+int(h/6),self.current[0]-int(w/6):self.current[0]+int(w/6),:]
                # Making a 'cross' signal to help choosing region
                zoom_in[int(h/6),int(4*w/30):int(19*w/120)] = np.uint8(zoom_in[int(h/6),int(4*w/30):int(19*w/120)]+125)
                zoom_in[int(h/6),int(21*w/120):int(6*w/30)] = np.uint8(zoom_in[int(h/6),int(21*w/120):int(6*w/30)]+125)
                zoom_in[int(4*h/30):int(19*h/120),int(w/6)] = np.uint8(zoom_in[int(4*h/30):int(19*h/120),int(w/6)]+125)
                zoom_in[int(21*h/120):int(6*h/30),int(w/6)] = np.uint8(zoom_in[int(21*h/120):int(6*h/30),int(w/6)]+125)

                # Transforming 'zoom_in' is a zoom (it is a crop right now)
                h_z, w_z = np.shape(zoom_in)[0],  np.shape(zoom_in)[1]
                zoom_in = cv2.resize(zoom_in[int(h_z/2)-int(h_z/4):int(h_z)-int(h_z/4), int(w_z/2)-int(w_z/4):int(w_z)-int(w_z/4)], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                image2[0:2*int(h/6),w-2*int(w/6):w]=zoom_in
                cv2.imshow(window_name, image2)
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                self.done = True
        cv2.destroyWindow(window_name)
        return self.points



def im2label(root, classes, **kwargs):
    '''Function to create a label mask from an image.
    
    This function creates another folder, with the same name of root plus
    the string "labels", and start to save the label images in this folder,
    with the same name. Since labeling takes lot of time, this function also 
    identify which images are not labeled before to start. Final output image
    is scaled from 0 to 255.
    
    im2label(root, classes)
    
    root: 'string'
        root directory where images are.
    
    classes: 'int'
        the number of classes to choose.
    
    **open_roi: 'string' (standard: None)
        If open_roi is not 'None', the algorithm choose open regions (regions
        that ends in the image bondary). If 'open_roi' = 'above', the chosen
        region will be an open region above the chosen area, the opposite
        happens if 'open_roi' = 'below', with a region below the chosen points.
    **cmap: 'int' (cv2 colormap)
    **show: 'boolean' (standard: False)
        If 'show' is True, show the final image and its label until user press
        'ESC', or any key.
    **equalize: 'boolean' (standard: False)
        If 'True', equalize grayscale image histogram.
    **color: 'toople' (standard: (200,200,200))
        Enter a different color to color the working line (R,G,B) with values
        from 0 to 255.
    
    Mouse actions:
    - left buttom: select a new point in the label;
    - right buttom: end selection and finish or go to another label;
    - ESC: finish selection (close the algorithm).
    
    **OBS1: When using 'open_roi', it is just possible to chose points from
    left part of the image to the right.
    OBS: The remaining unlabeled pixels will be set as the background pixels
    (they will belong to the last label)(if a label is chosen more than once,
    the last chosen label will be applied).
    OBS2: images can be multidimensional ([hiegth,width,dimensions])
    '''   
    # Transforming classes to integer
    classes = int(classes)
    # With 'kwargs' we can define extra arguments as input.
    cmap = kwargs.get('cmap')
    open_roi = kwargs.get('open_roi')
    show = kwargs.get('show')
    equalize = kwargs.get('equalize')
    color = kwargs.get('color')
    
    # If no color was chosen, choose gray:
    color = (200, 200, 200) if color==None else color

    # First, we create a folder with name 'root'+ ' labels', to save results
    os.chdir(root)
    os.chdir('..')
    basename = os.path.basename(os.path.normpath(root))
    # If folder has been already created, the use of try prevent error output
    try: os.mkdir(basename+' labels')
    except: pass
    os.chdir(basename+' labels')
    # Verify if folder has some already labaled images, if yes, skip the 
    # labeled ones
    image_names = os.listdir(root)
    if os.listdir(os.curdir):
        for name in os.listdir(os.curdir):
            if name in image_names:
                image_names.remove(name)
    # Shuffling the names
    shuffle(image_names)
    # This for will iterate in each image in 'root' folder
    for image_name in image_names:
        image = cv2.imread(os.path.join(root, image_name))
        # Discovering the image type [color or multidimensional (img_type1 = 3)
        # or gray (img_type1 = 2)]
        img_type1 = len(np.shape(image))
        if img_type1 == 2:
            img_type = 'gray'
            # Equalizing histogram
            if equalize == True:
                cv2.equalizeHist(image, image)
            image = image[..., np.newaxis]
            image[:,:,1] = image[:,:,0]
            image[:,:,2] = image[:,:,0]
            dim = 1
        else:
            img_type = 'color'
            dim = np.shape(image)[2]
        # first the image label will be a '-1' array
        if img_type == 'gray':
            label_image = np.zeros(np.shape(image), int)-np.ones(np.shape(image), int)
        else:
            label_image = np.zeros(np.shape(image[:,:,0]), int)-np.ones(np.shape(image[:,:,0]), int)
        # Image width and higher
        w = np.shape(image)[1]
        h = np.shape(image)[0]
        # Iterate in each label (except the last one, that is background)
        label = 1
        while label < classes:
            # The '.run' class gives the chosen points
            im2label = im2label_class()
            points = im2label.run(image, label, cmap, w, h, dim, img_type, color)
            points = np.asarray(points)
            # If no points were chosen, gives the option to label the unchosen
            # points in the image as background
            if len(points)<1:
                q1 = 'Label unchosen points as background?'
                q2 = '\n\n\'No\' will quit application (an error will eventually raised).'
                # Answer: asw = 6 (yes), asw = 7 (no) 
                asw = ctypes.windll.user32.MessageBoxW(0,q1+q2,"Pergunta", 4)
                if asw == 6:
                    break
            # Creating a roi to signaling the chosen region with '1'
            if img_type == 'gray':
                roi = np.zeros(np.shape(image), np.int32)
            else:
                roi = np.zeros(np.shape(image[:,:,0]), np.int32)
            # First we run when roi regions are closed (open_roi == None)
            if not open_roi:
                cv2.fillPoly(roi, [np.asarray(points)], (1, 1, 1))
                # Then we change the 'label_image' to 'label' when roi was chosen
                label_image[roi==1] = label
                # If is interesting to only change where there are no other
                # labels assigned, use another condition as follows:
                # label_image[(roi==1) & (label_image==-1)] = label
            elif open_roi == 'above' or open_roi == 'below':
                # If ROI is 'above', concatenate the upper image part, but
                # if the user choose points near to the sides, concatenate
                # the side last side points. Same to 'below'.
                if points[0,0] > h - points[0,1]: # DOWN-X
                    if w - points[-1,0] > h - points[-1,1]: # DOWN-DOWN
                        points[0,1] = h
                        points[-1,1] = h
                        if open_roi == 'above':
                            start_points = np.array([[w,h],[w,0],[0,0],[0,h]])
                        elif open_roi == 'below':
                            start_points = None
                    elif w - points[-1,0] > points[-1,1]: # DOWN-UP
                        points[0,1] = h
                        points[-1,1] = 0
                        if open_roi == 'above':
                            start_points = np.array([[0,0],[0,h]])
                        elif open_roi == 'below':
                            start_points = np.array([[w,0],[w,h]])
                    else: # DOWN-RIGHT
                        points[0,1] = h
                        points[-1,0] = w
                        if open_roi == 'above':
                            start_points = np.array([[w,0],[0,0],[0,h]])
                        elif open_roi == 'below':
                            start_points = np.array([[w,h]])
                            
                elif points[0,0] > points[0,1]: # UP-X
                    if w - points[-1,0] > h - points[-1,1]: # UP-DOWN
                        points[0,1] = 0
                        points[-1,1] = h
                        if open_roi == 'above':
                            start_points = np.array([[w,h],[w,0]])
                        elif open_roi == 'below':
                            start_points = np.array([[0,h],[0,0]])
                    elif w - points[-1,0] > points[-1,1]: # UP-UP
                        points[0,1] = 0
                        points[-1,1] = 0
                        if open_roi == 'above':
                            start_points = None
                        elif open_roi == 'below':
                            start_points = np.array([[w,0],[w,h],[0,h],[0,0]])
                    else: # UP-RIGHT
                        points[0,1] = 0
                        points[-1,0] = w
                        if open_roi == 'above':
                            start_points = np.array([[w,0]])
                        elif open_roi == 'below':
                            start_points = np.array([[w,h],[0,h],[0,0]])
                else: # LEFT-X
                    if w - points[-1,0] > h - points[-1,1]: # LEFT-DOWN
                        points[0,0] = 0
                        points[-1,1] = h
                        if open_roi == 'above':
                            start_points = np.array([[w,h],[w,0],[0,0]])
                        elif open_roi == 'below':
                            start_points = np.array([[0,h]])
                    elif w - points[-1,0] > points[-1,1]: # LEFT-UP
                        points[0,0] = 0
                        points[-1,1] = 0
                        if open_roi == 'above':
                            start_points = np.array([[0,0]])
                        elif open_roi == 'below':
                            start_points = np.array([[w,0],[w,h],[0,h]])
                    else: # LEFT-RIGHT
                        points[0,0] = 0
                        points[-1,0] = w
                        if open_roi == 'above':
                            start_points = np.array([[w,0],[0,0]])
                        elif open_roi == 'below':
                            start_points = np.array([[w,h],[0,h]])
                
                if start_points is not None:
                    points = np.concatenate((start_points,points), axis=0)
                cv2.fillPoly(roi, [np.asarray(points)], (1, 1, 1))
                # Assigning 'label' where 'roi' was chosen
                label_image[roi==1] = label
                # If is interesting to only change where there are no other
                # labels assigned, use another condition as follows:
                # label_image[(roi==1) & (label_image==-1)] = label
            else:
                raise ValueError('\nvariable \'open_roi\' has to be \'above\' or \'below\'')
            # Ask if the label was currectly chosen:
            q1 = 'Was label '+str(label)+' correctly chosen?'
            q2 = '\n\nSelect \'No\' to redo the labeling \n\n\'Yes\': to go forward..'
            # Answer: asw = 6 (yes), asw = 7 (no) 
            asw = ctypes.windll.user32.MessageBoxW(0,q1+q2,"Pergunta", 4)
            if asw == 6:
                # Ask if the user wants to choose more parts to the same label
                q1 = 'Want to choose more points for label '+str(label)+'?'
                q2 = '\n\nSelect \'No\' to continue.. \n\n\'Yes\': to select more labels.'
                # Answer: asw = 6 (yes), asw = 7 (no) 
                asw = ctypes.windll.user32.MessageBoxW(0,q1+q2,"Pergunta", 4)
                if asw == 7:
                    label += 1
        
        # Assigning the last label as background (label = 0).
        label_image[label_image==-1] = 0
        label_image = np.array(label_image, np.uint8)
        label_image = scale255(label_image)
        # If 'show' = True
        if show:
            fig, ax = plt.subplots(1,2)
            if img_type == 'color':
                ax[0].imshow(image[:,:,::-1])
            ax[0].axis('off')
            ax[1].imshow(label_image, vmax=np.max(label_image), vmin=np.min(label_image), cmap = 'viridis')
            ax[1].axis('off')
        # Adding final label image in 'label_images'
        cv2.imwrite(image_name, label_image)
    print('\nAll the images were labeled')



class improfile_class(object):
    '''This is a class to help improfile function (choose polygonal ROI)
    
    Read 'improfile' function for more informations.
    '''
    def __init__(self):
        self.done = False       # True when we finish the polygon
        self.current = (0, 0)   # Current position of mouse
        self.points = []        # The chosen vertex points
    
    # This function defines what happens when a mouse event take place.
    def mouse_actions(self, event, x, y, buttons, parameters):
        # If polygon has already done, return from this function
        if self.done:
            return
        # Update the mouse current position (to draw a line from last vertexn)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate a new vertex, so we add this to 'points' 
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        # Right buttom pressed indicate the end of drawing, so 'done = True'
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True
    
    # This function is to really 'run' the polyroi function
    def run(self, image, cmap, window_name, img_type, show):
        # If there is no a window name chose, apply the standard one.
        if window_name is None:
            window_name = "Choose a polygonal ROI"
        # Separating if we use or not a colormap.
        if cmap is not None:
            image_c = cv2.applyColorMap(image, cmap)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        # function to make the window with name 'window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(window_name, self.mouse_actions)
        
        # Defining thickness based on image size (to lines in image)
        image2 = image_c.copy()
        if np.shape(image2)[0] > 350:
            thickness = int(np.shape(image_c.copy())[0]/350)
        else:
            thickness = 1
        
        # loop to draw the lines while we are choosing the polygon vertices
        while(not self.done):
            # make a copy to draw the working line
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            
            # If at least 1 point was chosen, draw the polygon and the working
            # line. We use cv2.imshow to constantly show a new image with the
            # vertices already drawn and the updated working line
            if (len(self.points) > 0):
                cv2.polylines(image2, np.array([self.points]), False,
                              (200,200,200), thickness)
                cv2.line(image2, self.points[-1], self.current,
                         (200,200,200), thickness)
                cv2.imshow(window_name, image2)
            k = cv2.waitKey(50) & 0xFF
            if k == 27 or len(self.points) > 2:
                self.done = True
        length = np.hypot(self.points[0][0]-self.points[1][0],
                          self.points[0][1]-self.points[1][1])
        
        X = np.array(np.linspace(self.points[0][0],self.points[1][0],int(length)), int)
        Y = np.array(np.linspace(self.points[0][1],self.points[1][1],int(length)), int)
        profile = image2[X, Y]
        # When leaving loop, draw the polygon IF at least a point was chosen
        if (len(self.points) > 0):
            image3 = image.copy()
            cv2.polylines(image3, np.array([self.points]),False,(200,200,200))
        
        if show is not None:
            cv2.imshow(window_name, image3)
        
        return profile, self.points, window_name



def improfile(image, **kwargs):
    '''Find the profile of pixels intensity between two points in an image
    
    [profile, points_out] = improfile(image, **cmap, **window_name, **show)
    
    image: the input image
    points_in: input points to perform the profiling (if the points are already
    obtained).
    cmap: Chose the prefered colormap. If 'None', image in grayscale.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    window_name: choose a window name if you want to
    show: if True, it shows the image with the chosen polygon drawn
    profile: the output profile chosen
    points_out: the chosen vertices
    
    left buttom: select a new vertex
    right buttom: finish the vertex selection
    ESC: finish the function
    
    With this function it is possible to choose a polygonal ROI
    (region of interest) using the mouse. Use the left button to 
    choose te vertices and the right button to finish selection.
    
    **The arguments with double asteristic are optional (**kwargs).
    '''
    profileclass = improfile_class()
    
    # With 'kwargs' we can define extra arguments that the user can input.
    cmap = kwargs.get('cmap')
    window_name = kwargs.get('window_name')
    show = kwargs.get('show')
    
    # Discovering the image type [color (img_type1 = 3) or gray (img_type1=2)]
    img_type1 = len(np.shape(image))
    if img_type1 == 2:
        img_type = 'gray'
    else:
        img_type = 'color'
    
    # 'profileclass.run' actually run the program we need.
    [profile, points, window_name] = profileclass.run(image, cmap,
                                                  window_name, img_type, show)
    cv2.waitKey(500)    # To wait a little for the user to see the chosen ROI.
    cv2.destroyWindow(window_name)

    # to modify function:
    # # Next we find the profile. If points are already passed, just find the
    # # profile, if not, fint the points and profile with 'profileclass' class
    # if points_in is None:
    #     # 'profileclass.run' actually run the program we need.
    #     [profile, points, window_name] = profileclass.run(image, cmap,
    #                                                   window_name, img_type, show)
    #     cv2.waitKey(500)    # To wait a little for the user to see the chosen ROI.
    #     cv2.destroyWindow(window_name)
    # else:
    #     length = np.hypot(self.points[0][0]-self.points[1][0],
    #                       self.points[0][1]-self.points[1][1])
        
    #     X = np.array(np.linspace(self.points[0][0],self.points[1][0],int(length)), int)
    #     Y = np.array(np.linspace(self.points[0][1],self.points[1][1],int(length)), int)
    #     profile = image2[X, Y]
    
    return profile, np.asarray(points)



def scale255(image):
    '''
    Function to scale the image to the range [0,255]
    
    image_out = scale255(image)
    
    input:
    'image' (np.array): image to be rescaled
    
    output:
    image_out (np.array, np.uint8): rescaled image
    
    OBS: Image should be of shape (H, W, C), where H is the height, W is the
    width and C is the number of channels, for multichannel images.
    '''
    # Discovering if the image is grayscale or colorful
    if len(np.shape(image)) == 2:
        img_type = 'gray'
    # We use '>=3' to accounts for multichannel images
    elif len(np.shape(image)) >= 3:
        img_type = 'color'
    else:
        raise ValueError('Image has to be at least a 2D array, but '+
                         str(len(np.shape(image)))+'D array was given.')
    if img_type == 'gray':
        image = image - np.min(image)
        if np.max(image) != 0:
            image = image*(255/np.max(image))
    elif img_type == 'color':
        for n in range(len(np.shape(image))):
            image[:,:,n] = image[:,:,n] - np.min(image[:,:,n])
            if np.max(image[:,:,n]) != 0:
                image[:,:,n] = image[:,:,n]*(255/np.max(image[:,:,n]))
    else:
        raise ValueError('Image type was not recognized, see the'+
                         ' see the documentation for more information.')
    
    image = np.array(image, np.uint8)
    
    return image



class polyroi1(object):
    '''This is a class to help polyroi function (choose polygonal ROI)
    
    Read 'polyroi' function for more informations.
    '''
    def __init__(self):
        self.done = False       # True when we finish the polygon
        self.current = (0, 0)   # Current position of mouse
        self.points = []        # The choosen vertex points
    
    # This function defines what happens when a mouse event take place.
    def mouse_actions(self, event, x, y, buttons, parameters):
        # If polygon has already done, return from this function
        if self.done:
            return
        # Update the mouse current position (to draw a line from last vertexn)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate a new vertex, so we add this to 'points' 
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        # Right buttom pressed indicate the end of drawing, so 'done = True'
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True
    
    # This function is to really 'run' the polyroi function
    def run(self, image, cmap, window_name, img_type, show):
        # If there is no a window name chose, apply the standard one.
        if window_name is None:
            window_name = "Choose a polygonal ROI"
        # Separating if we use or not a colormap.
        if cmap is not None:
            image_c = cv2.applyColorMap(image, cmap)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        # function to make the window with name 'window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(window_name, self.mouse_actions)
        
        # Defining thickness based on image size (to lines in image)
        if np.shape(image_c)[0] > 500:
            thickness = int(np.shape(image_c)[0]/500)
        else:
            thickness = 1
        
        # loop to draw the lines while we are choosing the polygon vertices
        while(not self.done):
            # make a copy to draw the working line
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            
            # If at least 1 point was chosen, draw the polygon and the working
            # line. We use cv2.imshow to constantly show a new image with the
            # vertices already drawn and the updated working line
            if (len(self.points) > 0):
                cv2.polylines(image2, np.array([self.points]), False,
                              (200,200,200), thickness)
                cv2.line(image2, self.points[-1], self.current,
                         (255,255,255), thickness)
                cv2.imshow(window_name, image2)
            k = cv2.waitKey(50) & 0xFF
            if k == 27:
                self.done = True
        
        # When leaving loop, draw the polygon IF at least a point was chosen
        if (len(self.points) > 0):
            image3 = image.copy()
            cv2.fillPoly(image3, np.array([self.points]), (255,255,255))
            image4 = np.zeros(np.shape(image3), np.uint8)
            cv2.fillPoly(image4, np.array([self.points]), (255,255,255))
        
        if show is not None:
            cv2.imshow(window_name, image3)
        return image4, self.points, window_name



def polyroi(image, **kwargs):
    '''Choose a polygonhal ROI with mouse
    
    [image2, points] = polyroi(image, **cmap, **window_name, **show)
    
    image: the input image
    cmap: Chose the prefered colormap. If 'None', image in grayscale.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    window_name: choose a window name if you want to
    show: if True, it shows the image with the chosen polygon drawn
    image2: the output polygonal ROI
    points: the chosen vertices
    
    left buttom: select a new vertex
    right buttom: finish the vertex selection
    ESC: finish the function
    
    With this function it is possible to choose a polygonal ROI
    (region of interest) using the mouse. Use the left button to 
    choose te vertices and the right button to finish selection.
    
    **The arguments with double asteristic are optional (**kwargs).
    '''
    policlass = polyroi1()
    
    # With 'kwargs' we can define extra arguments that the user can input.
    cmap = kwargs.get('cmap')
    window_name = kwargs.get('window_name')
    show = kwargs.get('show')
    
    # Discovering the image type [color (img_type1 = 3) or gray (img_type1 =2)]
    img_type1 = len(np.shape(image))
    if img_type1 == 2:
        img_type = 'gray'
    else:
        img_type = 'color'
    
    # 'policlass.run' actually run the program we need.
    [image2, points, window_name] = policlass.run(image, cmap,
                                                  window_name, img_type, show)
    cv2.waitKey(500)    # To wait a little for the user to see the chosen ROI.
    cv2.destroyWindow(window_name)
    
    return image2, points



class crop_image1(object):
    '''This is a class to help the 'crop_image' function
    
    Read 'crop_image' for more informations.
    '''
    def __init__(self):
        self.done = False       # True when the crop area was already selected.
        self.current = (0, 0)   # Current mouse position.
        self.points = []        # Points to crop the image (corners).
    
    # This function defines what happens when a mouse event take place.
    def mouse_actions(self, event, x, y, buttons, parameters):
        # If cropping has already done, return from this function.
        if self.done:
            return
        # Update the mouse current position (to draw the working rectangle).
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate the first and the second main corners of
        # the rectangle to be chosen(upper-left and lower-right corners), so we
        # add this points to the 'points'.
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        # If right buttom pressed, we start to draw the rectangl again.
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points = []
    
    # This function actually run the 'crop_image' function
    def run(self, image, cmap, window_name, img_type):
        if cmap is not None:
            if img_type == 2:
                image_c = cv2.applyColorMap(image, cmap)
            else:
                image_c = image.copy()
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(window_name, image_c)
            cv2.waitKey(1) 
        # function to make the window with name 'window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(window_name, self.mouse_actions)
        
        # Defining thickness based on image size (to lines in image)
        thickness = int(np.shape(image_c)[1]/500)
        
        # Loop to draw the rectangles while the user choose the final one.
        while(not self.done):
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            
            # If at least 1 point was chosen, draw the working rectangle from
            # its upper-left or lower-right corner until the current mouse
            # position (working rectangle).
            if (len(self.points) > 0):
                if (len(self.points) == 1):
                    cv2.rectangle(image2, self.points[0], self.current,
                                  (200,200,200), thickness)
                    cv2.imshow(window_name, image2)
            # We creat k to exit the function if we pressee 'ESC' (k == 27)
            k = cv2.waitKey(50) & 0xFF
            if (len(self.points) > 1) or (k == 27):
                self.done = True
            # We have to use this for the case when right mouse button is pres-
            # sed. This is to stop shown the old erased rectangle (which we
            # erase with the right button click).
            if (len(self.points) == 0):
                cv2.imshow(window_name, image_c)
            
        cv2.destroyWindow(window_name)
        # When leaving loop, draw the final rectangle IF at least two
        # points were chosen.
        if (len(self.points) > 1):
            image3 = image.copy()
            cv2.rectangle(image3, self.points[0], self.points[1],
                          (200,200,200), thickness)
            
        return self.points, image3
    

def crop_image(image, **kwargs):
    '''Function to crop images using mouse
    
    [image2, points] = crop_image(image, **show, **cmap, **window_name)
    
    image: input image.
    show: 'True' to show the image with rectangle to be cropped. Otherwise, the
          image will not be shown.
    cmap: Chose the prefered colormap. If 'None', image in grayscale.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    window_name: choose a window name.
    image2: output imaga, with the cropping rectangle drown in it.
    points: a variable of type 'list' with the 2 main points to draw the crop-
            ping rectangle (upper-left and lower-right).
    
    How to use: 
        1. Left mouse button - choose the rectangle corners to crop the image
        (upper-left and lower-right). If two points were chosen, the rectangle
        will be completed and the function end.
        2. Right mouse click - erase the chosen points and starts the choose
        again from begening.
    '''
    # Discovering the image type [color (img_type = 3) or gray (img_type = 2)]
    img_type = len(np.shape(image))
    
    # Obtaining '**kwargs'
    cmap = kwargs.get('cmap')
    window_name = kwargs.get('window_name')
    show = kwargs.get('show')
    
    # If there is no a window name chose, apply the standard one.
    if window_name is None:
        window_name = "Choose a region to Crop"
    # Calling class:
    cropping_class = crop_image1()
    [points, image3] = cropping_class.run(image, cmap, window_name, img_type)
    
    # If 'show = True', show the final image, with the chosen cropping rectang.
    if show == True:
        window_name2 = 'Crop result'
        cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name2,image3)
    
    # Cropping image
    points1 = np.asarray(points[0]) # 'points' is a 'list' variable, so we used
    points2 = np.asarray(points[1]) # np.asarray to obtain an array of values.
    # Forming a vector with 4 points (x1, y1, x2, y2).
    points3 = np.concatenate((points1, points2))
    
    # Using rectng. to really crop the image. This 'IF' logic is stated because
    # we need to know the corners that were chosen, and its sequence.
    if points3[2] - points3[0] > 0:         # if x2 > x1 in (x1, y1),(x2,y2)
        if points3[3] - points3[1] > 0:     # if y2 > y1 in (x1, y1),(x2,y2)
            image2 = image[points3[1]:points3[3], points3[0]:points3[2]]
        else:                               # if y1 > y2 in (x1, y1),(x2,y2)
            image2 = image[points3[3]:points3[1], points3[0]:points3[2]]
    else:
        if points3[3] - points3[1] > 0:
            image2 = image[points3[1]:points3[3], points3[2]:points3[0]]
        else:
            image2 = image[points3[3]:points3[1], points3[2]:points3[0]]
    
    return image2, points



class crop_multiple1(object):
    
    def __init__(self):
        self.window_name = 'Choose the area to crop'    # Our window's name.
        self.done = False       # True when the crop area was already selected.
        self.current = (0, 0)   # Current mouse position.
        self.center = []        # Center point to crop the image.

    # This function defines what happens when a mouse event take place.    
    def mouse_actions(self, event, x, y , buttons, parameters):
        # If cropping has already done, return from this function.
        if self.done:
            return
        # If we have a mouse move, update the current position. We need this to
        # draw a 'working rectangle' when the user moves mouse (to show the
        # possible rectangle that the user could choose).
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate that the user chose the center point.
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.center.append((x, y))
    
    # This function actually run the 'crop_multiple' function
    def run(self, image, width, height, img_type, cmap):
        if cmap is not None:
            if img_type == 2:
                image_c = cv2.applyColorMap(image, cmap)
            else:
                image_c = image.copy()
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(self.window_name, image_c)
            cv2.waitKey(1) 
        # function to make the window with name 'self.window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(self.window_name, self.mouse_actions)
        
        # Loop to draw the 'working rect.' while the user choose the final one.
        while (not self.done):
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            cv2.imshow(self.window_name, image2)
            
            # The further logic implement a sum or a subtraction to width/2 and
            # height/2 from the central to obtain the rectangle corners. Exem-
            # ple: The upper-left corner is obtained subtracting width/2 and
            # height/2 from the central point coordinates.
            
            # We make 'width % 2' and 'height % 2' because 'number % 2' disco-
            # ver if its number is even or odd. And if one of this is odd, we
            # need to add 'one' when summing width/2 or height/2 to the center
            # point coordinates (otherwise the first chosen rectangle will not
            # macht with the other ones).
            if width % 2 == 0:
                if height % 2 == 0:
                    difference1 = tuple((np.int32(np.int(width/2)), np.int32(np.int(height/2))))
                    difference2 = tuple((np.int32(-np.int(width/2)), np.int32(-np.int(height/2))))
                else:
                    difference1 = tuple((np.int32(np.int(width/2)), np.int32(np.int(height/2)+1)))
                    difference2 = tuple((np.int32(-np.int(width/2)), np.int32(-np.int(height/2))))
            else:
                if height % 2 == 0:
                    difference1 = tuple((np.int32(np.int(width/2)+1), np.int32(np.int(height/2))))
                    difference2 = tuple((np.int32(-np.int(width/2)), np.int32(-np.int(height/2))))
                else:
                    difference1 = tuple((np.int32(np.int(width/2)+1), np.int32(np.int(height/2)+1)))
                    difference2 = tuple((np.int32(-np.int(width/2)), np.int32(-np.int(height/2))))
            
            # To find the rectang. corners we subtract 'width/2' and '-width/2'
            # from the 'central point' (or 'self.current', where the mouse is).
            point1 = np.subtract(self.current,difference1)
            point2 = np.subtract(self.current,difference2)
            point1 = tuple(point1)      # It was more easy to use tuple in the
            point2 = tuple(point2)      # function 'cv2.rectangle'.
            # Defining thickness based on image size
            thickness = int(np.shape(image2)[1]/500)
            cv2.rectangle(image2, point1, point2, (200,200,200), thickness)
            cv2.imshow(self.window_name, image2)
            
            # If a center point was already chosen (center > 0) or if the 'ESC'
            # key was pressed (k = 27), exit the 'while loop'.
            center = np.asarray(self.center)
            k = cv2.waitKey(50) & 0xFF
            if center.any() > 0 or k == 27:
                self.done = True
        
        # Using rect. to really crop the image. This 'IF' logic is stated be-
        # cause we need to know the corners that were chosen, and its sequence.
        if point2[0] - point1[0] > 0:         # if x2 > x1 in (x1, y1),(x2,y2)
            if point2[1] - point1[1] > 0:     # if y2 > y1 in (x1, y1),(x2,y2)
                image3 = image[point1[1]:point2[1], point1[0]:point2[0]]
            else:                             # if y1 > y2 in (x1, y1),(x2,y2)
                image3 = image[point2[1]:point1[1], point1[0]:point2[0]]
        else:
            if point2[1] - point1[1] > 0:
                image3 = image[point1[1]:point2[1], point2[0]:point1[0]]
            else:
                image3 = image[point2[1]:point1[1], point2[0]:point1[0]]
        
        # Closing this window we prevent that this window remain after running
        # this functino.
        cv2.destroyWindow(self.window_name)
        return image3, point1, point2



def crop_multiple(images, cmap):
    '''Function to crop multiple images with the same rectangle.
    
    [images2, points] = crop_multiple(images, cmap)
    
    images: input image (a'list' or a 'numpy.ndarray' variable). The I size has
            to be: 'I.shape = (n, heigh, width)', with 'n = number of images'.
    cmap: Chose the prefered colormap. If 'None', image in grayscale.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    images2: output cropped images (a 'numpy.ndarray' variable)
    points: a variable of type 'list' with the 2 main points to draw the crop-
            ping rectangle (upper-left and lower-right).

    First image:
        1. Left mouse button - choose the rectangle corners to crop the image
        (upper-left and lower-right). If two points were chosen, the rectangle
        will be completed and the function end.
        2. Right mouse click - erase the chosen points and starts the choose
        again from begening.
        
    Onother images:
        1. Move mouse to select where the center of cropping rectangle will be.
    
    This function uses mouse to choose a rectangle to crop in the first image,
    and uses the mouse again to place this same rectangle (same in dimentions)
    to crop the other images in a different place (different in terms of (x,y))
    '''
    # If image is a 'list' variable, we need to transform in a numpy array
    I = np.asarray(images)
    
    # Discovering the image type [color (img_type = 4) or gray (img_type = 3)]
    img_type = len(np.shape(images))
    
    # First image cropping uses the 'crop_image' function.
    [I00, points0] = crop_image(I[0,...], cmap=cmap)
    
    if img_type == 3:       # Grayscale images
        I2 = np.zeros((len(I), I00.shape[0], I00.shape[1]), np.uint8)
        I2[0,...] = I00
    elif img_type == 4:     # Color images
        I2 = np.zeros((len(I), I00.shape[0], I00.shape[1], I00.shape[2]), np.uint8)
        I2[0,...] = I00
    
    # Here we create 'lists' to put the points that are the rectangle corners.
    pointA = []
    
    pointA.append((min(points0[0][0],points0[1][0]), min(points0[0][1],points0[1][1])))
    
    # Taking 'points' from a 'list' variable to a 'numpy array' one.
    points1 = np.asarray(points0[0])
    points2 = np.asarray(points0[1])
    points3 = np.concatenate((points1, points2))    # concatenation points.
    
    # With the points information, we can obtain the width and height
    width = abs(points3[2] - points3[0])
    height = abs(points3[3] - points3[1])
    
    # For loop to perform a crop in all the images in 'I'
    for n in range(1,len(I)):
        # The best practice is every time call the class before use its functi.
        crop_class = crop_multiple1()
        [I2[n,...], point1, point2] = crop_class.run(I[n], width, height,
                                                     img_type, cmap)
        pointA.append((min(point1[0],point2[0]), min(point1[1],point2[1])))
    
    return I2, pointA



class crop_poly_multiple1(object):
    
    def __init__(self):
        self.window_name = 'Choose the area to crop'    # Our window's name.
        self.done = False       # True when the crop area was already selected.
        self.current = (0, 0)   # Current mouse position.
        self.center = []        # Center point to crop the image.

    # This function defines what happens when a mouse event take place.    
    def mouse_actions(self, event, x, y , buttons, parameters):
        # If cropping has already done, return from this function.
        if self.done:
            return
        # If we have a mouse move, update the current position. We need this to
        # draw a 'working polygon' when the user moves mouse (to show the
        # possible polygon that the user could choose).
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        # Left buttom pressed indicate that the user chose the center point.
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.center.append((x, y))
    
    # This function actually run the 'crop_multiple' function
    def run(self, image, points, equalize, img_type, cmap, show, window_name):
        self.window_name = window_name
        if equalize == True:
            image = cv2.equalizeHist(image)
        if cmap is not None:
            if img_type == 'gray':
                image_c = cv2.applyColorMap(image, cmap)
            else:
                image_c = image.copy()
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.window_name, image_c)
            cv2.waitKey(1)
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            image_c = image.copy()
            cv2.imshow(self.window_name, image_c)
            cv2.waitKey(1) 
        # function to make the window with name 'self.window_name' starts
        # to interact with the user by mouse, acording 'self.mouse_actions'
        cv2.setMouseCallback(self.window_name, self.mouse_actions)
        
        # Loop to draw the 'working polygon' while user choose the final one.
        points1 = np.array(points,dtype='int32')
        while (not self.done):
            # THIS COPY OF ORIGINAL IMAGE FORCE THAT ONLY ONE RECTANGLE BE
            # DRAWN IN EACH IMAGE PLOT. Try to comment and see what happens.
            image2 = image_c.copy()
            cv2.imshow(self.window_name, image2)
            
            # We have to sum the polygon position with the current mouse one:
            current1 = np.array(self.current,dtype='int32')
            
            # The next few lines is to read the user keyboard. We needed to de
            # fine something global ('global key1') to take this value outside
            # the class.
            def on_press(key):
                global key1
                key1 = key
                return False
            
            def on_release(key):
                global key1
                key1 = 'None'
                return False
            # To turn on the listener who read the keyboard
            listener = keyboard.Listener(on_press=on_press,
                                         on_release=on_release)
            listener.start() 
            time.sleep(0.05) # waiting few miliseconds.
            
            # This is to bring points1 to the center where mouse is.
            points1[:,0] = points1[:,0] - np.mean(points1[:,0]) + current1[0]
            points1[:,1] = points1[:,1] - np.mean(points1[:,1]) + current1[1]
            
            # If user use 'up' and 'down' arrows, the ROI will rotate more
            # then if the user press 'left' or 'right' arros.
            try:
                if str(key1) == 'Key.up':
                    points1 = rotate2D(points1,current1,ang=+np.pi/16)
                elif str(key1) == 'Key.down':
                    points1 = rotate2D(points1,current1,ang=-np.pi/16)
                elif str(key1) == 'Key.left':
                    points1 = rotate2D(points1,current1,ang=+np.pi/64)
                elif str(key1) == 'Key.right':
                    points1 = rotate2D(points1,current1,ang=-np.pi/64)
                else:
                    pass
            except: pass
            
            # To use 'polylines', we have to enter with a 'int32' array with
            # a specific shape, then we use 'reshape'
            points1 = np.array(points1,np.int32)
            # This is to make the shape aceptable in 'cv2.polylines' and 
            # 'cv2.fillConvexPoly'
            points1 = points1.reshape((-1, 1, 2))
            
            # Defining line thicknesse based on image size
            if np.shape(image2)[0] > 600:
                thickness = int(np.shape(image2)[0]/300)
            else:
                thickness = 2
            
            cv2.polylines(image2,[points1],True,(200,200,200), thickness)
            cv2.imshow(self.window_name, image2)
            points1 = np.squeeze(points1, axis=1)
            # If a center point was already chosen (center > 0) or if the 'ESC'
            # key was pressed (k = 27), exit the 'while loop'.
            center = np.asarray(self.center)
            k = cv2.waitKey(50) & 0xFF
            if center.any() > 0 or k == 27:
                self.done = True
        
        points1 = points1.reshape((-1, 1, 2))
        image3 = np.zeros(np.shape(image),np.uint8)
        cv2.fillConvexPoly(image3, np.array(points1,dtype='int32'),
                           (255,255,255))
        # Closing this window we prevent that this window remain after running
        # this function (if show is not True).
        if show is not True:
            cv2.destroyWindow(self.window_name)
        # Since we change the shape of 'points1', we need to use np.squeeze.
        return image3, np.squeeze(points1, axis=1)



def crop_poly_multiple(images, **kwargs):
    '''Function to crop multiple images with the same polygon.
    
    [images2, points] = crop_poly_multiple(images, cmap)
    
    images: input image (a'list' or a 'numpy.ndarray' variable). The I size has
            to be: 'I.shape = (n, heigh, width)', with 'n = number of images'.
    cmap: Chose the prefered colormap. Use 'None' for color or grayscale image.
          Some colormap exemples: cv2.COLORMAP_PINK, cv2.COLORMAP_HSV,
          cv2.COLORMAP_PARULA, cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW, etc.
    equalize: When choosen 'True', the showed image has histogram equalization.
    show: When choosen 'True', the final image with polyroi is showed.
    window_name: You can choose the window name.
    images2: output cropped images (a 'numpy.ndarray' variable)
    points: a 'list' variable with the first point selected in each polygon.

    First image:
        1. Left mouse button - to choose the polygon corners.
        2. Right mouse click - to finish.
        
    Onother images:
        1. Move mouse to select where the center of cropping rectangle will be.
    
    This function uses mouse to choose a polygon to crop in the first image,
    and uses the mouse again to place this same polygon (same in dimentions)
    to crop the other images in a different place (different in terms of (x,y))
    '''
    # If image is a 'list' variable, we need to transform in a numpy array
    I = np.asarray(images)

    # Obtaining '**kwargs'
    cmap = kwargs.get('cmap')
    window_name = kwargs.get('window_name')
    show = kwargs.get('show')
    equalize = kwargs.get('equalize')
    
    # If there is no a window name chose, apply the standard one.
    if window_name is None:
        window_name = "Choose a region to Crop"
    
    # Discovering the image type [color (img_type1 = 3), gray (img_type1 = 4)]
    img_type1 = len(np.shape(images))
    if img_type1 == 3:
        img_type = 'gray'
    else:
        img_type = 'color'
    
    # First image cropping uses the 'polyroi' function.
    I2 = []
    if equalize == True:
        I[0,...] = cv2.equalizeHist(I[0,...])
    [Itemp, points] = polyroi(I[0,...], cmap = cmap, window_name = window_name)
    I2.append(Itemp)
    # First saved point.
    pointA = []
    pointA.append(np.asarray(points,np.int32))
    
    # For loop to perform a crop in all the images in 'I'
    for n in range(1,len(I)):
        # The best practice is every time call the class before use its functi.
        crop_class = crop_poly_multiple1()
        [Itemp, points1] = crop_class.run(I[n], points, equalize,
                                          img_type, cmap, show, window_name)
        I2.append(Itemp)
        pointA.append(points1)
    
    return I2, pointA



def filter_finder(x, y, **kwargs):
    '''
    Function to study which filter to use for a signal y(x)
    
    output = filter_finder(x, y, **kwargs)
    
    inputs:
    show (string): operation mode, to choose what the function will plot. If
    'time' is chosen, this function compare the original signal with filtered
    one in the time domain (or in whatherver domain the original signal is). If
    'frequency' is chosen, the signal and filter spectra are compared. If
    'interpolation' is chosen, the original signal is compared with the inter-
    polated one for visualization. Standard value is 'time'.
    N (int): the number of points to be used in FFT and 'x' interpolation.
    fs (int): frequency of sampling
    Wn (int): critical frequency
    order (int): filter order
    
    output:
    output (tuple): Depends on the mode. If the mode is 'frequency', return w
    (wavelengths) and h (frequency response, to be ploted e.g. as np.abs(h)).
    If the mode is 'time', it returns the x and the y for the filtered signal.
    
        
    If show = 'frequency' compare signal and filter in the frequency domain.
    If show = 'time' compare signal and filtered signal in the time domain.
    If show = 'interpolation', it compares the true signal with its interpola-
    tion (to see if interpolation is good for this signal).
    '''
    N = kwargs.get('N')
    fs = kwargs.get('fs')
    Wn = kwargs.get('Wn')
    order = kwargs.get('order')
    show = kwargs.get('show')
    window = kwargs.get('window')
    
    if N is None:
        N = 2**10
    if fs is None:
        fs = 1
    if Wn is None:
        Wn = 0.25
    if order is None:
        order = 5
    if window is None:
        window = signal.windows.hamming(N)
    if show is None:
        show = 'time'
    # Defining 'T' as 1 over the sampling frequency 'fs'
    T = 1/fs
    # First interpolating the signal
    f = interp1d(x, y)
    x_ip = np.linspace(x[0], x[-1], N, endpoint=True)
    y_ip = f(x_ip)
    if show == 'interpolation':
        plt.subplots()
        plt.plot(x, y)
        plt.plot(x_ip, y_ip, 'o')
    # Calculating the FFT of the signal y(x)
    y_w = y_ip*window
    yf = fft(y_w)
    yf_log = 20*np.log10(np.abs(yf[0:N//2])/np.max(np.abs(yf[0:N//2])))
    xf = fftfreq(N, T)[:N//2]
    # Plotting the freq. response of the signal y(x)
    if show == 'frequency':
        plt.subplots()
        plt.plot(xf, yf_log)
    # Calculating the IIR filter for this signal
    sos = signal.iirfilter(order, Wn=Wn, fs=fs, btype='low', ftype='butter',
                           output='sos')
    w, h = signal.sosfreqz(sos, worN=N, fs=fs)
    if show == 'frequency':
        plt.plot(w, 20*np.log10(np.abs(h)), label='Filter')
        plt.show()
        return (w, h)
    if show == 'time':
        y_filt = signal.sosfiltfilt(sos, y)
        plt.subplots()
        plt.plot(x, y, label='Original signal')
        plt.plot(x, y_filt, label='Filtered signal')
        plt.legend()
        plt.show()
        return (x, y_filt)



def highpass_gauss(img, **kwargs):
    '''
    Apply a Gaussian-based high-pass filter
    
    I = highpass_gauss(img, **kwargs)
    
    'img' (np.array): image to be filtered (2D np.array)
    'sigma' (float): sigma from Gaussian function to be used in the filtering
    'bkg' (int): integer to be used as a background
    return (np.array): filtered image, rescaled to the 0-255 range.
    
    This functino apply a Gaussian-based high-pass filter by subtracting a
    gaussian filtered image from the original one (resulting in the high-freq.
    content). This function is very useful, and the first choise for testing
    high-pass filters in a broad range of samples.    
    '''
    ksize = kwargs.get('ksize')
    sigma = kwargs.get('sigma')
    bkg = kwargs.get('bkg')
    if ksize is None:
        ksize = (11,11)
    if sigma is None:
        sigma = 15
    if bkg is None:
        bkg = 127
    
    return scale255(img - cv2.GaussianBlur(img, ksize, sigma))



def highpass_fft(img, **kwargs):
    '''
    Apply a high-pass FFT filter in a 2D image
    
    I = highpass_fft(img, **kwargs)
    
    'image' (np.array): image to be filtered (2D np.array)
    'frac' (float): fraction from the image to be removed from the low-frequen-
    cy part of the spectrum (standard: 0.25) (if passed a value greater than
    0.5, the function will assign 'frac = 0.5').
    'prnt' (boolean): true to print the images that yield from processing for
    testing purpose (standard: False).
    'title' (string): test to be added as a title in the printed images if
    'prnt = True'
    'return' (np.array): return the final image, after FFT filtering, rescaled
    to the 0-255 range.
    
    This function passes a high-pass filter based on eliminating a fraction of
    the FFT image spectrum. The fraction to be eliminated is stated in 'frac'
    and accounts for half of the spectrum. so if you choose 'frac=0.5' the
    whole spectrum will be eliminated, and no image will return. Note that the
    returned image is rescaled to the 0-255 range.
    
    **kwargs indicates the inputs that can be passed to the function, but are
    not mandatory.    
    '''
    frac = kwargs.get('frac')       # Fraction to be removed from spectrum
    prnt = kwargs.get('prnt')       # To print the images from the processing
    title = kwargs.get('title')     # To add a title to the printed images
    if frac is None:
        frac = 0.25
    elif frac > 0.5:
        frac = 0.5
    if prnt is None:
            prnt = False
    if title is None:
        title = None
    # First we calculate the bidimensional FFT of input image
    im_fft = fftpack.fft2(img)
    
    if prnt:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(np.log(np.abs(im_fft)))
        ax[0].axis('off')
        plt.suptitle(title)
    # Here we remove the parts from low frequencies, by vanishing its content
    r, c = im_fft.shape
    im_fft[0:int(frac*r),:] = 0
    im_fft[r-int(frac*r):r,:] = 0
    im_fft[:,0:int(frac*c)] = 0
    im_fft[:,c-int(frac*c):c] = 0
    
    if prnt:
        ax[1].imshow(np.log(np.abs(im_fft)))
        ax[1].axis('off')
    # Then we calculate the inverse FFT of transformed image
    final = fftpack.ifft2(im_fft).real
    
    if prnt:
        ax[2].imshow(final)
        ax[2].axis('off')
        plt.subplots_adjust(top=0.98, bottom=0.08, left=0.01,
                            right=0.98, hspace=0.01, wspace=0.01)
    
    return scale255(final)



def lowpass_fft(img, **kwargs):
    '''
    Apply a low-pass FFT filter in a 2D image
    
    I = lowpass_fft(img, **kwargs)
    
    'image' (np.array): image to be filtered (2D np.array)
    'frac' (float): fraction from the image to be removed from the high-
    frequency part of the spectrum (standard: 0.25) (if passed a value greater
    than 0.5, the function will assign 'frac = 0.5').
    'prnt' (boolean): true to print the images that yield from processing for
    testing purpose (standard: False).
    'title' (string): test to be added as a title in the printed images if
    'prnt = True'
    'return' (np.array): return the final image, after FFT filtering, rescaled
    to the 0-255 range.
    
    This function passes a low-pass filter based on eliminating a fraction of
    the FFT image spectrum. The fraction to be eliminated is stated in 'frac'
    and accounts for half of the spectrum. so if you choose 'frac=0.5' the
    whole spectrum will be eliminated, and no image will return. Note that the
    returned image is rescaled to the 0-255 range.
    
    **kwargs indicates the inputs that can be passed to the function, but are
    not mandatory. 
    '''
    frac = kwargs.get('frac')       # Fraction to be removed from spectrum
    prnt = kwargs.get('prnt')       # To print the images from the processing
    title = kwargs.get('title')     # To add a title to the printed images
    if frac is None:
        frac = 0.25
    elif frac > 0.5:
        frac = 0.5
    if prnt is None:
            prnt = False
    if title is None:
        title = None
    # First we calculate the bidimensional FFT of input image
    im_fft = fftpack.fft2(img)
    
    if prnt:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(np.log(np.abs(im_fft)))
        ax[0].axis('off')
        plt.suptitle(title)
    # Here we devide by 'frac' to use 'frac' as a fraction o which will be re-
    # moved
    r, c = im_fft.shape
    im_fft[int(0.5*r-frac*r):int(0.5*r+frac*r),:] = 0
    im_fft[:,int(0.5*c-frac*c):int(0.5*c+frac*c)] = 0
    
    if prnt:
        ax[1].imshow(np.log(np.abs(im_fft)))
        ax[1].axis('off')
    # Then we calculate the inverse FFT of transformed image
    final = fftpack.ifft2(im_fft).real
    
    if prnt:
        ax[2].imshow(final)
        ax[2].axis('off')
        plt.subplots_adjust(top=0.98, bottom=0.08, left=0.01,
                            right=0.98, hspace=0.01, wspace=0.01)
    
    return scale255(final)



def filt_hist(hist):
    '''
    Functino to 'filter' equalized histogram, removing the zeros values 
    between positive ones (util to process equalized historgrams).
    
    output = filt_hist(hist)
    
    input:
    hist (np.array): input histogram to be 'filtered' (removed the zero values)
    
    output:
    output (np.array): output histogram without zero values between positive
    ones.
    '''
    output = np.zeros(np.shape(hist))
    output[0] = hist[0]
    # We go until the middle of the signal (len(hist)//2)
    active = False
    for n in range(1, len(hist)//2):
        if (hist[n-1]>0) and (hist[n]==0):
            output[n] = hist[n-1]
            active = True
        if hist[n]>0:
            active = False
        if active:
            output[n] = output[n-1]
        else:
            output[n] = hist[n]
    # Then we come back from the end to the middle, to do not make the final
    # zero values from the spectrum becoming non-zero values by this function.
    output[-1] = hist[-1]
    active = False
    for n in range(len(hist)-2, len(hist)//2-1, -1):
        if (hist[n+1]>0) and (hist[n]==0):
            output[n] = hist[n+1]
            active = True
        if hist[n]>0:
            active = False
        if active:
            output[n] = output[n+1]
        else:
            output[n] = hist[n]
    return output



def imroiprop(I):
    """Function to get properties of a ROI in a image
    
    [props, Imask] = imroiprop(I)
    
    props[0]: sum all pixel values in ROI;
    props[1]: mean of non-zero values in ROI;
    props[2]: std of non-zero values in ROI.
    """
    # First we choose a polygonal ROI:
    [Imask, points] = polyroi(I, cmap = cv2.COLORMAP_PINK)
    
    # The mask poits of ROI came with "255" value, but we need the value "1".
    Imask[Imask > 0] = 1
    
    # Preparing a vector to receive variables:
    props = np.zeros(3, np.double)
    
    # Multiplying by mask
    Itemp = I*Imask
    # Integrating all the pixel values:
    props[0] = np.sum(Itemp)
    # Mean pixel value from ROI:
    props[1] = Itemp[Itemp!=0].mean()
    # Standar deviation from pixels in ROI:
    props[2] = Itemp[Itemp!=0].std()
    
    return props, Imask



def imchoose(images, cmap):
    '''Function to chose images of given image set.
    
    chosen = imchoose(images, cmap)
    
    images: input images (a'list' or a 'numpy.ndarray' variable). The I shape
            has to be: 'np.shape(I)=(n, heigh, width)', with 'n' being the
            number of images.
    cmap: Chose the prefered pyplot colormap (it is a 'string' variable). Use
          'None' for color or grayscale image. Some colormap exemples: pink,
          CMRmap, gray, RdGy, viridis, terrain, hsv, jet, etc.
    chosen: A 'np.int' column vector with '1' for a chosen image, and '0'
            otherwise, in the column position corresponpding the image
            position. OBS: 'len(chosen) = len(images)'.

    How it works:
        Click in the image number that you want to choose. The image will
        chang, and will appear '(chosen)'. After choose all images, press
        'enter' or 'esc' key.
    OBS: this function create a object named 'key1'!
    '''
    # If image is a 'list' variable, we need to transform in a numpy array
    I = np.asarray(images)
    
    done = False    
    chosen = np.squeeze(np.zeros([1,len(I)]),axis=0)
    # Calling figure before printing the images
    fig, ax = plt.subplots(1,len(I))
    for n in range(0,len(I)):
        ax[n].imshow(I[n], cmap)
        ax[n].axis('off')
    # Next 'adjust'  to show the images closer.
    plt.subplots_adjust(top=0.976,bottom=0.024,left=0.015,right=0.985,
                        hspace=0.0,wspace=0.046)
    while(not done):
        # The next few lines is to read the user keyboard. We needed to de
        # fine something global ('global key1') to take this value outside
        # the class 'Listener'.
        def on_press(key):
            global key1
            key1 = key
            # When we recurn 'False', the listener stops, we always stops be-
            # cause otherwise the key always is the pressed key (looks like the
            # user is pressing the key continously, what affects the logic).
            return False
        
        def on_release(key):
            return False
        # To turn on the listener who read the keyboard
        listener = keyboard.Listener(on_press=on_press,
                                     on_release=on_release)
        listener.start() 
        # The 'plt.pause' allow the 'pyplot' have time to show the image.
        plt.pause(2) # waiting few miliseconds.
        # If user press 'esc' or 'enter', the program closes
        try:
            # If user press 'enter' or 'esc', the program ends.
            if str(key1) == 'Key.esc':
                done = True
            elif str(key1) == 'Key.enter':
                done = True
            # The user will chose pressing a key number (1,2,3,etc). This has
            # to add the value '1' (that means image chosen) to 'chosen' in the
            # position 'int(temp)-1', because remember Python starts in '0'.
            if str(key1) is not None and len(str(key1)) < 7:
                temp = np.mat(str(key1))
                chosen[int(temp)-1] = 1
                
                if chosen.any() > 0:
                    # Plotting the image
                    for n in range(0,len(I)):
                        if chosen[n] == 1:
                            # We change I[n] to become a little white:
                            Itemp = np.uint8(255*((I[n]+50)/np.max(I[n]+50)))
                            ax[n].imshow(Itemp, cmap)
                            ax[n].axis('off')
                            ax[n].set_title('(Escolhida)')
                        else:
                            ax[n].imshow(I[n], cmap)
                            ax[n].axis('off')
                    plt.subplots_adjust(top=0.976,bottom=0.024,left=0.015,
                                        right=0.985,hspace=0.0,wspace=0.046)
                    plt.pause(2)
        except: pass
    plt.close(fig)
    return np.array(chosen,np.int)



def align_features(I1, I2, draw):
    ''' Align images with Feature Based algorithm, from OpenCV
    
    [Ir, warp] = align_features(I1, I2, draw)
    
    Parameters
    ----------
    I1 : numerical-array (grayscale)
        Image to be aligned (array).
    I2 : numerical-array (grayscale)
        Reference image.
    draw: if 'True', an image with the chosen features is ploted.

    Returns
    -------
    Ir : numpy-array
        Aligned image.
    warp : 3x3 numpy-array
        Warping matrix.

    '''
    MAX_FEATURES = 500       # Make  **kwargs
    GOOD_MACH_PERCENT = 0.20 # Make  **kwargs
    
    # Detect ORB features and compute descriptors:
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(I1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(I2, None)
    
    # Match features:
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score:
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Removing some matches:
    numGoodMatches = int(len(matches)*GOOD_MACH_PERCENT)
    matches = matches[:numGoodMatches] # choose the good ones
    # matches = matches[len(matches)-numGoodMatches:] # choose not so good ones
    
    # Drawing the classified matches:
    if draw == True:
        imMatches = cv2.drawMatches(I1,keypoints1, I2,keypoints2, matches,None)
        plt.subplots()
        plt.imshow(imMatches)
        plt.axis('off')
    
    # Extracting location of good matches:
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for n, match in enumerate(matches):
        points1[n, :] = keypoints1[match.queryIdx].pt
        points2[n, :] = keypoints2[match.trainIdx].pt
    
    # Finding homography that align this two images
    warp, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Using homography
    height, width = I1.shape[0], I1.shape[1]
    Ir = cv2.warpPerspective(I1, warp, (width, height))
    
    return Ir, warp



def align_ECC(images, warp_mode):
    '''Thi function align a set of gray imagess using a function from 'OpenCV'
    
    [Ia, warp_matrix] = align_ECC(images, warp mode)
    
    images: these are the input images, with all the images that we want to
            align, respecting the standard images shape '(n, widt, height)',
            where 'n' is the image number (n.max() = number of images).
    
    warp_mode: the transformation needed in the form: cv2.MOTION_TRANSF, subs
               tituing 'TRANSF' by: TRANSLATION, EUCLIDEAN, AFFINE, HOMOGRAPHY.
    
    Ia: aligned images with shape = (n, height, width), being n = number of
        images.
    
    '''
    # Chossing the kind of image motion (image warp, or warp mode)
    if warp_mode == None:
        warp_mode = cv2.MOTION_TRANSLATION
    
    # Transforming in numpy array (from a possible 'tuple' or 'list')
    images1 = images.copy()
    images2 = np.asarray(images1)
    
    # Creating the warp matrix, the matrix with transformation coefficients, to
    # trnsform a missalined image in a aligned one (or to perform ANY transfor-
    # mation in a given image/matrix. At first, it will be a identity matrix.
    # In the HOMOGRAPHY mode, we need a 3x3 matrix, otherwise, a 2x3.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix0 = np.eye(3, 3, dtype = np.float32)
    else:
        warp_matrix0 = np.eye(2, 3, dtype = np.float32)
                    
    n_iterations = 500     # Maximum number of iterations.
    epsilon = 1e-5 # Threshold increment of correlation coef. betwn iterations
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iterations,
                epsilon)
    
    # To allocate a 'warp_matrix1' for every image, in a new array called warp_
    # matrix1, of shape = (n, warp_matrix1(x), warp_matrix1(y)), with n = number
    # of images. P.S.: warp_matrix1[0] is not used.
    warp_shape = warp_matrix0.shape
    warp_matrix1 = np.zeros((len(images2),warp_shape[0],warp_shape[1]))
       
    # This for is to find the transformation that make one image into
    # a aligned one (a 3x3 matrix). In this case, the algorithm used
    # is ECC, described in the paper (DOI: 10.1109/TPAMI.2008.113).
    for n in range(1,len(images2)):
        (cc, warp_matrix1[n,...]) = cv2.findTransformECC(images2[n,...],
                                                         images2[0,...],
                                                         warp_matrix0,
                                                         warp_mode, criteria)
#                                                         inputMask = None,
#                                                         gaussFiltSize = 5)
    
    # We will need the size of 'I' images.
    size = images2.shape[1:]
    
    # Loop to apply transformations chosen (with ECC) for all images.
    Ia = np.zeros((len(images2), size[0], size[1]), np.uint8)
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    for n in range(1,len(images2)):
        warp_matrix2 = warp_matrix1[n,...]
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            Ia[n,...] = cv2.warpPerspective(images2[n,...], warp_matrix2,
                                             (size[1], size[0]), flags)
        else:
            Ia[n,...] = cv2.warpAffine(images2[n,...], warp_matrix2,
                                        (size[1], size[0]), flags)
    
    # The first aligned image is the original one (the reference image).
    Ia[0,...] = images2[0,...]
    
    # Defining the output warp matrix (will be = warp_matrix1, but without the
    # first empty part (warp_matrix1[0,:,:], that is always zero))
    warp_matrix = warp_matrix1[1:,:,:]
                
    return Ia, warp_matrix



def imwarp(images, warp_matrix):
    '''This function warp (using cv2.warpAffine) a given image (or a set of
    images) using the input warp matrix:
        
    images_output = imwarp(images, warp_matrix)
    
    OBS: This 'warp_matrix' is iqual to that 'warp_matrix' output from 
    # 'align_ECC' function.
    '''
    # To assing a value to variable 'size', we need to assing zero in it.
    size = np.zeros(2, np.uint64)
    
    numb = []
    # Discovering the size of images. If len(np.shape(images)) = 2, we have
    # only one image to warp, if it is = 3, we  have a set of images to warp.
    if len(np.shape(images)) == 2:
        size[0] = np.shape(images)[0]
        size[1] = np.shape(images)[1]
        numb = 1                        # number of images to warp.
    elif len(np.shape(images)) == 3:
        size[0] = np.shape(images)[1]
        size[1] = np.shape(images)[2]
        numb = np.shape(images)[0]      # number of images to warp.
        
    # This flag is necessario to warp an image.
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    
    if numb == 1:
        # Defining the size/shape of the output image (or images).
        images_output = np.zeros((size[0], size[1]), np.uint8)
        warp_matrix2 = warp_matrix[0,...]
        images_output = cv2.warpAffine(images, warp_matrix2,
                                       (size[1], size[0]), flags)
    else:
        images_output = np.zeros((numb, size[0], size[1]), np.uint8)
        # This loop is to warp all the images we have.
        for n in range(0, numb+1):
            warp_matrix2 = warp_matrix[n,...]
            images_output = cv2.warpAffine(images[n], warp_matrix2,
                                           (size[1], size[0]), flags)
    
    return images_output



def isoareas(folder, numb, **kwargs):
    '''Integrate pixels in isoareas defined by the mouse.
    
    [F, Fi] = isoareas(folder, numb)
    
    folder: folder in which there are the images we want to process.
    numb: the number of isoareas you want to.
    beep: if equal True, we have a beep after each image processed.
    
    F[n,0]: tumor thickness of isoarea number 'n'
    F[n,1]: mean pixel value of isoarea number 'n'
    F[n,2]: standard deviation for all pixels in isoarea number 'n'.
    F[n,3]: mode for pixel values of isoarea 'n'
    F[n,4]: median of pixel values for isoarea 'n'
    Fi: 'F' interpolated with a constant distance step of 0.83*10^-6.
    
    This program is used to calculate the mean and the standar deviation for
    the pixels finded in the intersection between a ROI (region of interest)
    and the isoareas chosen by user. The isoareas will be chosen by the user by
    the left and right isolines that delimitat it.
    
    First: choose (with left button clicks in mouse) the points that delimit
    the first isoline, in the side of epidermis.
    
    Second: choose the points that delimit the last isoline, on the side of
    dermis (inner part).
    
    Third: choose a ROI (region of interest) in which the pixel values will be
    evaluated (the pixel values will be evaluated inside this ROI for each iso-
    area calculated, I mean: ther will be caluculated the pixel values for the
    intersection between each isoarea with the chosen ROI).
    
    Fourth: choose a line in the 'depth' direction (to show to the program what
    is depth).
    '''
    # First, we load all the images
    I = load_gray_images(folder, -1)
    
    # Obtaining the pixel size of each pixel
    pixel_size = kwargs.get('pixel_size')
    
    # Uncomment if you need a beep
    beep = kwargs.get('beep')
    
    # If the user doesn't enter with a pixel_size value, make 'pixel_size=1'.
    if pixel_size == None:
        pixel_size = 1
    
    # Creating empty lists for fluorecence ('F') and output images ('Iout').
    F = []
    Iout = []
    
    # Then we start a loop to calculate for all images in 'folder'.
    for m in range(0,len(I)):
        # First, we extract the 'm' image from folder.
        I1 = np.asarray(I[m])
        
        # Just to show the begining image
        fig, ax = plt.subplots()
        im = ax.imshow(I1, cmap = "pink")
#        ax.title('This is I1 image')
        fig.colorbar(im)
        plt.show()
        
        # Choosing the ROI of the first function.
        window = 'Choose a ROI for the side of epidermis (outside)'
        [Itemp, points1] = polyroi(I1, cmap = cv2.COLORMAP_PINK,
                                   window_name = window)
        points1 = np.asarray(points1)  # 'polylines' function require in array.
        
        # Drawing the chosen points in the begining image.
        I2 = I1.copy()
        cv2.polylines(I2, [points1], False, (220,200,200), 3)
        
        # Choosing the ROI of the second function.
        window = 'Choose a ROI for the side of dermis (inner part)'
        [Itemp, points2] = polyroi(I2, cmap = cv2.COLORMAP_PINK,
                                   window_name = window)
        points2 = np.asarray(points2)  # 'polylines' function require in array.
        points2[0,:] = points1[0,:]
        points2[-1,:] = points1[-1,:]
        
        
        # Drawing the chosen points in an image, and plotting them.
        I3 = I2.copy()
        cv2.polylines(I3, [points2], False, (200,200,220), 3)
        plt.subplots()
        plt.imshow(I3)
        plt.title('This is I3 image')
        
        # If we mantain the 'x' and 'y' of 'points', it will not be a function,
        # because for the same value 'x' we have more than 1 value of 'y'. So
        # we change 'x' and 'y' positions in 'points'
        points1a = np.zeros([len(points1[:,0]),len(points1[0,:])], np.double)
        points2a = np.zeros([len(points2[:,0]),len(points2[0,:])], np.double)
        points1a[:,1] = points1[:,0]
        points1a[:,0] = points1[:,1]
        points2a[:,1] = points2[:,0]
        points2a[:,0] = points2[:,1]
        
        plt.subplots()
        plt.plot(points1a[:,0],points1a[:,1])
        plt.plot(points2a[:,0],points2a[:,1])
        plt.title('points1a[:,0],points1a[:,1]\npoints2a[:,0],points2a[:,1]')
        
        # In the next steps we'll interpolate the points we have.
        div = 0.5       # Defining the increment between 'x' values.
        points1b = np.zeros([np.int(abs(points1a[-1,0]-points1a[0,0])/div),2],
                             np.double)
        points2b = np.zeros([np.int(abs(points1a[-1,0]-points1a[0,0])/div),2],
                             np.double)
        f_points1a = interpolate.interp1d(points1a[:,0],points1a[:,1])
        f_points2a = interpolate.interp1d(points2a[:,0],points2a[:,1])
        points1b[:,0] = np.arange(min(points1a[0,0], points1a[-1,0]),
                                  max(points1a[0,0], points1a[-1,0]), div)
        points2b[:,0] = points1b[:,0]
        points1b[:,1] = f_points1a(points1b[:,0])
        points2b[:,1] = f_points2a(points2b[:,0])
        
        
        # We have to put all the isolines in one variable (better to handle).
        isolines = np.zeros([np.int(abs(points1a[-1,0]-points1a[0,0])/div),
                             2*numb+2], np.double)
        # The first line is the 'points1b' that we already defined.
        isolines[:,0:2] = points1b[:,0:2]
        # The last line is the 'points2b' that we already defined.
        isolines[:,-2:] = points2b[:,0:2]
        
        # Here we find the isolines between the borders.
        for n in range(0,numb-1):
            isolines[:,2+n*2] = points1b[:,0]
            isolines[:,3+n*2] = points1b[:,1]+(n+1)*((points2b[:,1]-points1b[:,1])/numb)
        
        # Plotting the isolines
        plt.subplots()
        plt.title('isolines[:,0+n*2], isolines[:,1+n*2]')
        for n in range(0,numb+1):
            plt.plot(isolines[:,0+n*2], isolines[:,1+n*2], linewidth = 3, color = "white")
        plt.show
        
        # Changing, again, 'x' with 'y' to plot the isolines in the figure.
        isolines2 = np.zeros([np.int(abs(points1a[-1,0]-points1a[0,0])/div),
                             2*numb+2], np.double)
        for n in range(0,numb+1):
            isolines2[:,0+n*2] = isolines[:,1+n*2]
            isolines2[:,1+n*2] = isolines[:,0+n*2]
        
        # Plotting the isolines
        plt.subplots()
        plt.title('isolines2[:,0+n*2], isolines2[:,1+n*2]')
        for n in range(0,numb+1):
            plt.plot(isolines2[:,0+n*2], isolines2[:,1+n*2], linewidth = 3, color = "white")
        plt.show
        
        # The next steps create the images with the ROIs of isolines. 'I4' is
        # only to user see, and 'I5' is what we'll use.
        I4 = []
        I5 = []
        # Defining the variable that will help us to "paint" the image 'I4'.
        factor = np.round(250/numb)
        if factor > 10:
            factor = 10
        # This two "for's" is actually to create the images.
        Itemp1 = np.zeros([np.shape(I1)[0],np.shape(I1)[1]], np.uint8)
        for n in range(0,numb):
            pts_temp = np.concatenate((isolines2[:,0+n*2:2+n*2],
                                       isolines2[::-1,2+n*2:4+n*2]))
            pts_temp = np.matrix.round(pts_temp)
            pts_temp = np.array(pts_temp, np.int)
            cv2.fillPoly(Itemp1, [pts_temp],
                         (factor*(n+1),factor*(n+1),factor*(n+1)))
        for n in range(0,numb):
            Itemp2 = np.zeros([np.shape(I1)[0],np.shape(I1)[1]], np.uint8)
            pts_temp = np.concatenate((isolines2[:,0+n*2:2+n*2],
                                       isolines2[::-1,2+n*2:4+n*2]))
            pts_temp = np.matrix.round(pts_temp)
            pts_temp = np.array(pts_temp, np.int)
            cv2.fillPoly(Itemp2, [pts_temp], (1,1,1))
            I5.append(Itemp2)
        
        # Transforming 'I4' and 'I5' from list to 'numpy array'.
        I4 = I1.copy() + ((0.25*np.max(I1))/np.max(Itemp1))*Itemp1
        I4 = np.matrix.round(I4)
        I4 = np.array(I4, np.uint8)
        I5 = np.asarray(I5)
        
        plt.subplots()
        plt.imshow(I4)
        plt.axis('off')
        plt.title('Plotting I4 image')
        
        im, ax = plt.subplots(1,numb)
        for n in range(0,numb):
            ax[n].imshow(I5[n])
            ax[n].axis('off')
        plt.title('Plotting all I5 images')
        
        # Chosing the region where fluorescence will be calculated (fluorescen-
        # ce will be calculated inside this region ('Imask'), in each isoarea).
        window = 'Choose the region in which fluorescence will be calculated'
        [Imask, points3] = polyroi(I4, cmap = cv2.COLORMAP_PINK,
                                             window_name = window)
        # We need '1' and '0' pixels values to be the ROI that will multiply
        # our isoareas.
        Imask[Imask > 0] = 1
        
        
        # In order to calculate the real width between the isoareas, we need to determine
        # the direction of depth (a vector from outside of skin, entering in the tumor
        # deeper regions). This vector will be drawn with next 'polyroi'
        window = 'Choose a vector entering in tumor (determine what is depth)'
        [Itemp, points4] = polyroi(I1, cmap = cv2.COLORMAP_PINK, window_name = window)
        # Extracting the first two points fron 'points4'.
        depth_line = np.array([np.asarray(points4[0]), np.asarray(points4[1])], np.int)
        # Then we find the coefficients of a line curve 'ax+b'.
        coefficients = np.polyfit(depth_line[:,0], depth_line[:,1], 1)
        
        # Now we create 2 images, one is the sum of all isoareas (Itemp1), and
        # other one is the sum of all isoareas times the chosen ROI (Itemp2).
        vect1 = []
        vect2 = []
        for n in range(0,len(I5)):
            Itemp1 = np.zeros((np.shape(I1)), np.uint8)
            Itemp2 = np.zeros((np.shape(I1)), np.uint8)
            Itemp1 = I1*I5[n]
            Itemp2 = I1*I5[n]*Imask
         
            # Now we define a line pointing the direction of the 'depth', and 
            # see when this line intercept I1*I5*Imask.
            line = np.zeros((int(len(I1[0,:])/0.1),2), np.double)
            line[:,0] = np.arange(0,len(I1[0,:]),0.1)
            vector1 = []    # Pixels that belong to 'line' and to 'I5*Imask'
            vector2 = []    # Pixels that belong to 'line' and to 'I5*Imask
            for k in range(0,int(len(I1[:,0]))):
                line[:,1] = coefficients[0]*line[:,0] + k
                Itemp = np.zeros((np.shape(I1)), np.uint8)
                # Defining the points to draw.
                points5 = np.asarray(np.round((line[0,:],line[-1,:])), np.int)
                cv2.polylines(Itemp, [points5], False,(1,1,1),1)
                value1 = np.count_nonzero(Itemp*Itemp1)
                value2 = np.count_nonzero(Itemp*Itemp2)
                if value1 > 0:             # If it is true, 'line' intercept I1*I5.
                    if value1 == value2:    # If it is true, 'line' intercept I1*I5*Imask.
                        vector1.append(value2)
                        vector2.append(0)
                    elif value1 > value2 and value2 > 0: # If true, 'line' intercept 
                        vector1.append(0)             # I1*I5*Imask, but not in all the
                        vector2.append(value2)        # pixels belonging to I1*I5
                    elif value1 > value2 and value2 == 0:
                        vector1.append(0)           # If true, 'line'
                        vector2.append(0)           # does not intercept nothing.
                else:
                    vector1.append(0)
                    vector2.append(0)
           
            vector1 = np.asarray(vector1)
            vector2 = np.asarray(vector2)
            vect1.append(vector1)
            vect2.append(vector2)
        
        vect1 = np.asarray(vect1)
        vect2 = np.asarray(vect2)
        
        # The next lines is to find the mean of distance between isoareas:
        mean1 = []
        for n in range(0,len(vect1[:,0])):
            sum_temp = vect1[n,:].sum()
            num_temp = np.count_nonzero(vect1[n,:])
            mean_temp = sum_temp/num_temp
            mean1.append(mean_temp)
        mean2 = np.mean(np.array(mean1))
        # Finally, defining the mean width, in 'meters' for each isoarea. The
        # term 'np.sqrt((1+coef[0]**2))' means that when we have diagonal
        # pixels, we have greater distances, it comes from line equation.
        width = np.sqrt((1+coefficients[0]**2))*pixel_size*mean2
        plt.subplots()
        for n in range(0,len(vect1[:,0])):
            plt.plot(vect1[n,:])
#            plt.plot(vect2[n,:])
        
        # Calculating fluorescence.
        Iout0 = []
        F0 = np.zeros([numb,5], np.double)
        for n in range(0,numb):
            Itemp = I1*I5[n]*Imask
            Iout0.append(Itemp)
            F0[n,0] = n*width + width/2
            F0[n,1] = Itemp[Itemp!=0].mean()
            F0[n,2] = Itemp[Itemp!=0].std()
            F0[n,3] = stats.mode(Itemp[Itemp!=0], axis = None)[0]
            F0[n,4] = np.median(Itemp[Itemp!=0])
        F.append(F0)
        Iout0 = np.asarray(Iout0)
        Iout.append(Iout0)
        
        # If we need a beep:
        if beep == True:
            frequency = 2500  # Set Frequency To 2500 Hertz
            duration = 300    # Set Duration To 1000 ms == 1 second
            for n in range(0,3):
#                time.sleep(0.0005)
                winsound.Beep(frequency, duration)
        
    F = np.asarray(F)
    
    # In order to find an interpolated function for 'F' with equal steps in 
    # depth (being depth = F[n][:,m]), we define 'Fi', the interpolated 'F'.
    Fi = []
    for n in range(0,len(F)):
        # To know the number of points we do the next lines:
        pixel_width = 0.83e-6
        number_temp = np.int((F[n][-1,0] - F[n][0,0])/pixel_width)
        Fi_temp = np.zeros([number_temp, 2], np.double)
        Fi_temp[:,0] = np.arange(F[n][0,0], F[n][0,0]+number_temp*pixel_width,
                                 pixel_width)
        # Defining the function with 'interp1d'.
        function_Fi = interpolate.interp1d(F[n][:,0], F[n][:,1])
        Fi_temp[:,1] = function_Fi(Fi_temp[:,0])
        Fi.append(Fi_temp)
    
    
    # To save all the data, we first enter in the folder where te images are.
    os.chdir(folder)
    path = 'results'
    if os.path.exists(path) is not True:
        os.mkdir(path)
    os.chdir(path)
    
    # First, we save the 'txt' files
    path1 = 'txt-files'
    if os.path.exists(path1) is not True:
        os.mkdir(path1)
    os.chdir(path1)
    # Transforming 'F' from 'list' to 'numpy array'.
    
    # Actually saving the files.
    for n in range(0,len(F)):
        np.savetxt('F'+str(n)+'.txt', F[n])
        np.savetxt('Fi'+str(n)+'.txt', Fi[n])
    
    # Second, we save the 'numpy' files.
    os.chdir(folder)
    os.chdir(path)
    path2 = 'numpy-files'
    if os.path.exists(path2) is not True:
        os.mkdir(path2)
    os.chdir(path2)
    # Actually saving the files.
    for n in range(0,len(F)):
        np.save('F'+str(n), F[n])
        np.save('Fi'+str(n), Fi[n])
    
    return F, Fi



def good_colormaps(image):
    '''This function show a list of the good 'matplotlib.pyplot' colormaps:\n
    prism
    terrain
    flag ** (a lot of contrast)
    pink *
    coolwarm
    nipy_spectral
    gist_stern **
    gist_ncar
    Spectral
    hsv
    jet
    CMRmap (article)
    viridis (parula) (article)
    gnuplot
    RdGy ***
    BrBG
    
    image = input image
    
    The program will output a sequence of three figures showing the colormaps
    '''
    name1 = "Normal published colormaps4"
    plt.figure(name1)
    plt.subplot(131).set_title('gray')
    plt.imshow(image, cmap = "gray")
    plt.xticks([]), plt.yticks([])
    plt.subplot(132).set_title('viridis')
    plt.xticks([]), plt.yticks([])
    plt.imshow(image, cmap = "viridis")
    plt.subplot(133).set_title('CMRmap')
    plt.xticks([]), plt.yticks([])
    plt.imshow(image, cmap = "CMRmap")
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()  
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,
                        hspace=0.15, wspace=0.15)
    
    name2 = "Different Colormaps 14"
    plt.figure(name2)
    plt.subplot(231).set_title('prism')
    plt.imshow(image, cmap = "prism")
    plt.xticks([]), plt.yticks([])
    plt.subplot(232).set_title('terrain')
    plt.imshow(image, cmap = "terrain")
    plt.xticks([]), plt.yticks([])
    plt.subplot(233).set_title('flag')
    plt.imshow(image, cmap = "flag")
    plt.xticks([]), plt.yticks([])
    plt.subplot(234).set_title('gist_stern')
    plt.imshow(image, cmap = "gist_stern")
    plt.xticks([]), plt.yticks([])
    plt.subplot(235).set_title('pink')
    plt.imshow(image, cmap = "pink")
    plt.xticks([]), plt.yticks([])
    plt.subplot(236).set_title('nipy_spectral')
    plt.imshow(image, cmap = "nipy_spectral")
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,
                        hspace=0.15, wspace=0.15)
    
    name3 = "Different Colormaps 24"
    plt.figure(name3)
    plt.subplot(231).set_title('gist_ncar')
    plt.imshow(image, cmap = "gist_ncar")
    plt.xticks([]), plt.yticks([])
    plt.subplot(232).set_title('RdGy')
    plt.imshow(image, cmap = "RdGy")
    plt.xticks([]), plt.yticks([])
    plt.subplot(233).set_title('gnuplot')
    plt.imshow(image, cmap = "gnuplot")
    plt.xticks([]), plt.yticks([])
    plt.subplot(234).set_title('gist_stern')
    plt.imshow(image, cmap = "gist_stern")
    plt.xticks([]), plt.yticks([])
    plt.subplot(235).set_title('hsv')
    plt.imshow(image, cmap = "hsv")
    plt.xticks([]), plt.yticks([])
    plt.subplot(236).set_title('BrBG')
    plt.imshow(image, cmap = "BrBG")
    plt.xticks([]), plt.yticks([])
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95,
                        hspace=0.15, wspace=0.15)
