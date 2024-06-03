# -*- coding: utf-8 -*-
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

import matplotlib.pyplot as plt
import imfun1 as imfun
import numpy as np



file_path = r'H:\Drives compartilhados\Terapia Fotodinâmica e Fluorescência\Imagens\Confocal\2024.05.15 - Dianeth Sara\CONFOCAL\Creme1h\2C1P_1.lsm'

full_image = imfun.read_lsm(file_path)


plt.subplots()
plt.imshow(full_image)
plt.axis('off')
plt.tight_layout()
plt.show()
