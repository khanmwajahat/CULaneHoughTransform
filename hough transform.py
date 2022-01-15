#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import cv2
from matplotlib import pyplot as plt


# In[3]:


import os
from os.path import basename

test_images_dir = "C:/Users/kahnw/Desktop/Mcmaster/neural networks/Project Reports/test set 1/"
image_names = os.listdir(test_images_dir)
if ".DS_Store" in image_names: image_names.remove(".DS_Store")
label_names = [os.path.splitext(name)[0].lower() for name in image_names]
image_names = [test_images_dir + name for name in image_names]

def show_image_list(image_list, cols=2, fig_size=(15, 15), image_labels=label_names):
    image_count = len(image_list)
    rows = image_count / cols
    plt.figure(figsize=fig_size)
    for i in range(0, image_count):
        image_name = image_labels[i]

        plt.subplot(rows, cols, i+1)
        img = image_list[i]
        cmap = None
        if len(img.shape) < 3:
            cmap = "gray"
        plt.title(basename(image_name))    
        plt.imshow(img, cmap=cmap)
    plt.tight_layout()
    plt.show()

test_images = [mpimg.imread(image_name) for image_name in image_names]


# In[4]:


def remove_noise(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
def discard_colors(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)
def region_of_interest(image, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(image)
   
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are non-zero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines
def weighted_image(image, initial_image, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_image, α, image, β, λ)
def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
def weighted_img(img, initial_img, α=0.7, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
            





# In[5]:


def pipeline(original_image):
    ysize, xsize = original_image.shape[:2]
    bottom_left = (120, ysize)
    top_left = (750, 318)
    top_right = (850, 318)
    bottom_right = (900, ysize)
    vertices = np.array([[bottom_left , top_left, top_right, bottom_right]], dtype=np.int32)
    image2=remove_noise(original_image,3)
    image3=discard_colors(image2)
    image4=detect_edges(image3,150,250)
    xsize = image3.shape[1]
    ysize = image3.shape[0]
    dx1 = int(0.003 * xsize)
    dx2 = int(0.8 * xsize)
    dy = int(0.1* ysize)
    print(dx1,dx2,dy)
    # calculate vertices for region of interest
    vertices = np.array([[(dx1, ysize), (dx2, dy), (xsize - dx2, dy), (xsize - dx1, ysize)]], dtype=np.int32)
    image5 = region_of_interest(image4, vertices)
    rho = 0.8
    theta = np.pi/180
    threshold = 20
    min_line_len = 10
    max_line_gap = 20
    lines = hough_lines(image5, rho, theta, threshold, min_line_len, max_line_gap)
    lines_image = np.zeros((*image4.shape, 3), dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            after_transform=cv2.line(image5, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=5)
    lines_image=region_of_interest(after_transform, vertices)
    line_image = np.copy((original_image)*0)
    draw_lines(line_image, lines, thickness=3)
    line_image = region_of_interest(line_image, vertices)
    final_image = weighted_image(line_image, original_image)
    return final_image
show_image_list([pipeline(image) for image in test_images])
    
    
    


# In[ ]:


image=cv2.imread('C:/Users/kahnw/Desktop/Mcmaster/neural networks/Project Reports/test set 1/00135.jpg')
cv2.imshow('image window', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


image2=remove_noise(image,3)
cv2.imshow('image window', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[14]:


image3=discard_colors(image2)
plt.imshow(image3, cmap="gray")
plt.show()


# In[15]:


image4=detect_edges(image3,150,250)
plt.imshow(image4,cmap='gray')
plt.show()


# In[16]:


xsize = image3.shape[1]
ysize = image3.shape[0]
dx1 = int(0.003 * xsize)
dx2 = int(0.8 * xsize)
dy = int(0.1* ysize)
print(dx1,dx2,dy)
# calculate vertices for region of interest
vertices = np.array([[(dx1, ysize), (dx2, dy), (xsize - dx2, dy), (xsize - dx1, ysize)]], dtype=np.int32)
image5 = region_of_interest(image4, vertices)
plt.imshow(image5,cmap='gray')
plt.show()


# In[28]:


rho = 0.8
theta = np.pi/180
threshold = 20
min_line_len = 100
max_line_gap = 200
lines = hough_lines(image5, rho, theta, threshold, min_line_len, max_line_gap)
lines_image = np.zeros((*image5.shape, 3), dtype=np.uint8)
for line in lines:
        for x1,y1,x2,y2 in line:
            after_transform=cv2.line(image4, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=5)


# In[29]:


plt.imshow(after_transform)
plt.show()


# In[19]:


lines_image=region_of_interest(after_transform, vertices)
plt.imshow(after_transform)
plt.show()


# In[20]:



line_image = np.copy((image)*0)
draw_lines(line_image, lines, thickness=3)
line_image = region_of_interest(line_image, vertices)
final_image = weighted_image(line_image, image)


# In[21]:


cv2.imshow('image window', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




