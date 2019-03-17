# -*- coding: UTF-8 -*-  
#廖沩健
#自动化65
#2160504124

from __future__ import division, print_function
import numpy as np
import cv2
import math


SOBEL_MASK = np.array([
[[-1, -2, -1],
[0, 0, 0],
[1, 2, 1]], 
[[-1, 0, 1],
[-2, 0, 2],
[-1, 0, 1]]])

LAPLACIAN_MASK = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])


def load_file(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return img


def median_filter(img, kernel_size):
    h = img.shape[0]
    w = img.shape[1]
    new_img = np.zeros_like(img).astype(np.uint8)
    
    def find_median_value(x,y):
        bound = int((kernel_size - 1) / 2)
        value_list = []
        for i in range(-bound, bound+1):
            if x+i < 0 or x+i > h - 1:
                continue
            for j in range(-bound, bound+1):
                if y+j < 0 or y+j > w - 1:
                    continue
                value_list.append(img[x+i,y+j])
        return np.median(np.array(value_list))

    for i in range(h):
        for j in range(w):
            new_img[i,j] = find_median_value(i,j)
    return new_img


def gaussian_kernel(size, sigma=1.5):
    kernel = np.zeros((size,size), dtype=np.float)
    offset = int((size - 1) / 2)
    for i in range(size):
        for j in range(size):
            kernel[i,j] = np.exp(-((i-offset)**2 + (j-offset)**2)/2/sigma**2)
    kernel /= (2*math.pi*sigma)
    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_filter(img, kernel):
    h = img.shape[0]
    w = img.shape[1]
    new_img = np.zeros_like(img).astype(np.uint8)

    def get_patch(x, y):
        patch = np.zeros_like(kernel).astype(np.uint8)
        bound = int((kernel.shape[0] - 1) / 2)
        for i in range(-bound, bound+1):
            for j in range(-bound, bound+1):
                if x+i < 0 or y+i < 0 or x+i > h - 1  or y+i > w - 1:
                    p = 0
                else:
                    p = img[x+i, y+i]
                patch[i+1, j+1] = p        
        return patch

    for i in range(h):
        for j in range(w):
            patch = get_patch(i, j)
            new_img[i,j] = np.sum(np.multiply(kernel, patch))
    return new_img


def unsharpen_mask(img, coeff=1):
    kernel = gaussian_kernel(3)
    blur = gaussian_filter(img, kernel)
    # blur = cv2.GaussianBlur(img, (3,3), 1.5, 1) 
    mask = img - blur
    img += coeff*mask
    # img = tmp.astype(np.uint8)
    return img


def sobel(img, mask):
    gradient = np.zeros_like(img).astype(np.uint8)
    h = img.shape[0]
    w = img.shape[1]

    def get_patch(x, y):
        patch = np.zeros((3,3)).astype(np.uint8)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if x+i < 0 or y+i < 0 or x+i > h - 1  or y+i > w - 1:
                    p = 0
                else:
                    p = img[x+i, y+i]
                patch[i+1, j+1] = p        
        return patch

    for i in range(h):
        for j in range(w):
            patch = get_patch(i,j)
            gx = abs(np.sum(np.multiply(patch, mask[0])))
            gy = abs(np.sum(np.multiply(patch, mask[1])))
            gradient[i,j] = gx + gy
    
    return gradient


def laplacian(img, mask):
    img = cv2.GaussianBlur(img, (3,3), 1.5, 1)
    gradient = np.zeros_like(img).astype(np.uint8)
    h = img.shape[0]
    w = img.shape[1]

    def get_patch(x, y):
        patch = np.zeros((3,3)).astype(np.uint8)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if x+i < 0 or y+i < 0 or x+i > h - 1  or y+i > w - 1:
                    p = 0
                else:
                    p = img[x+i, y+i]
                patch[i+1, j+1] = p        
        return patch

    for i in range(h):
        for j in range(w):
            patch = get_patch(i,j)
            g2 = np.sum(np.multiply(patch, mask))
            gradient[i,j] = np.clip(g2, 0, 255)
    
    return gradient


def plot_1(name, method, size):
    img = load_file(name)
    if method == 'median':
        res = median_filter(img, size)
    elif method == 'gaussian':
        kernel = gaussian_kernel(size)
        res = gaussian_filter(img, kernel)
    elif method == 'gaussian_standard':
        res = cv2.GaussianBlur(img,(size,size),1.5)
    else:
        raise NameError

    cv2.imwrite('./res/{}{}x{}.bmp'.format(name+str('_')+method, size, size), res)
    print('finish {} {} {}x{}'.format(name, method, size, size))


def plot_2(name, method):
    img  = load_file(name)
    if method == 'unshapen':
        res = unsharpen_mask(img, coeff=2)
    elif method == 'sobel':
        res = sobel(img, SOBEL_MASK)
    elif method == 'laplacian':
        res = laplacian(img, LAPLACIAN_MASK)
    elif method == 'canny':
        res = cv2.Canny(img, 30, 200)
    else:
        raise NameError

    cv2.imwrite('./res/{}.bmp'.format(name+str('_')+method), res)
    print('finish {} {}'.format(name, method))



if __name__ == '__main__':

    plot_1('test1.pgm', 'median', 3)
    plot_1('test1.pgm', 'median', 5)  
    plot_1('test1.pgm', 'median', 7)  

    plot_1('test2.tif', 'median', 3)  
    plot_1('test2.tif', 'median', 5)
    plot_1('test2.tif', 'median', 7)
    
    plot_1('test1.pgm', 'gaussian', 3)  
    plot_1('test1.pgm', 'gaussian', 5)  
    plot_1('test1.pgm', 'gaussian', 7)

    plot_1('test2.tif', 'gaussian', 3)  
    plot_1('test2.tif', 'gaussian', 5)  
    plot_1('test2.tif', 'gaussian', 7)

    plot_2('test3_corrupt.pgm', 'unshapen')
    plot_2('test3_corrupt.pgm', 'sobel')
    plot_2('test3_corrupt.pgm', 'laplacian')
    plot_2('test3_corrupt.pgm', 'canny')

    plot_2('test4.tif', 'unshapen')
    plot_2('test4.tif', 'sobel')
    plot_2('test4.tif', 'laplacian')
    plot_2('test4.tif', 'canny')
    




