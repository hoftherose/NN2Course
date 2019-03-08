import imageio
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

def grayscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def dodge(front,back):
    result=front*255/(255-back) 
    result[np.logical_or(result > 255, back ==255)] =255
    return result.astype('uint8')

def sketchify(img, sigma):
    gray_img = grayscale(img)
    gray_inv_img = 255-gray_img
    blur_img = scipy.ndimage.filters.gaussian_filter(gray_inv_img,sigma=sigma)
    final_img = dodge(blur_img,gray_img)
    return final_img
