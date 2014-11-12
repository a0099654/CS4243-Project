import cv2
import cv2.cv as cv
import numpy as np
import sys

# reload image as color image for paiting corner point
image1 = cv2.imread("project_background_grass_sky_trees_533_400.png", cv2.CV_LOAD_IMAGE_COLOR)
image = cv2.imread("project_background_grass_sky_trees_533_400.png", cv2.CV_LOAD_IMAGE_COLOR)
image2 = cv2.imread('frame_f--25_g_0_h_0.jpg', cv2.CV_LOAD_IMAGE_COLOR)

th = 5
w1, h1, b1 = image1.shape
w2, h2, b2 = image2.shape

wd = int((w1-w2)/2)
hd = int((h1-h2)/2)

print wd, hd, w2, h2

image[100+wd:100+wd+w2, hd:hd+h2,:] = image2[0:w1-100,:,:]
print image[wd, hd, :]

for i in range(wd, wd+w2):
    for j in range(hd, hd+h2):
        if(image[i,j,0] < th and image[i,j,1] < th and image[i,j,2] < th):
            image[i,j,:] = image1[i,j,:]

cv2.imwrite('result.jpg', image)
    
