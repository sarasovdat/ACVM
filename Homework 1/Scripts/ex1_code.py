import numpy as np 
import cv2

from ex1_utils import *

# ASSIGNMENT 1 : Lucas-Kanade optical flow


sig = 1

def lucaskanade(im1 , im2 , N):
    """
    Implementation of a simple (non-pyramidal) Lucas-Kanade optical flow estimation method.

    Args:
        im1 (matrix): first image matrix (grayscale)
        im2 (matrix): second image matrix (grayscale)
        N (int): size of the neighborhood (NxX)
    """
    
    # (1) Odvodi:    
    im1x, im1y = gaussderiv(im1, 1)
    im2x, im2y = gaussderiv(im2, 1)
    It = gausssmooth(im2 - im1, 1)
    
    # Pixel-wise average
    Ix = (im1x + im2x) / 2
    Iy = (im1y + im2y) / 2
    
    # (2) Mnozenja:   
    Ixt = Ix * It 
    Iyt = Iy * It 
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy 
    
    # (3) u, v 
    # https://towardsdatascience.com/basics-of-kernels-and-convolutions-with-opencv-c15311ab8f55
    
    kernel = np.ones((N, N))
    
    Ixt_sum = cv2.filter2D(Ixt, -1, kernel)
    Iyt_sum = cv2.filter2D(Iyt, -1, kernel)
    Ixx_sum = cv2.filter2D(Ixx, -1, kernel)
    Iyy_sum = cv2.filter2D(Iyy, -1, kernel)
    Ixy_sum = cv2.filter2D(Ixy, -1, kernel)
    
    D = (Ixx_sum * Iyy_sum) - (Ixy_sum) ** 2
    D[abs(D) < 10e-9] = 10e-3
    
    u = - ((Iyy_sum * Ixt_sum - Ixy_sum * Iyt_sum) / D)
    v = - ((Ixx_sum * Iyt_sum - Ixy_sum * Ixt_sum) / D)
    
    return u, v


def hornschunck(im1, im2, n_iters, lmbd):
    """
    Implementation of Horn-Schunck algorithm.

    Args:
        im1 (matrix): first image matrix (grayscale)
        im2 (matrix): first image matrix (grayscale)
        n_iters (int): number of iterations
        lmbd (float): parameter
    """
    
    # (1) Odvodi:    
    im1x, im1y = gaussderiv(im1, 1)
    im2x, im2y = gaussderiv(im2, 1)
    It = gausssmooth(im2 - im1, 1)
    
    # Pixel-wise average
    Ix = (im1x + im2x) / 2
    Iy = (im1y + im2y) / 2
    
    # (2) Mnozenja:   
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    
    Ld = np.array([[0.0, 0.25, 0.0],
                   [0.25, 0.0, 0.25],
                   [0.0, 0.25, 0.0]])
    
    
    # Pripravimo u in v
    
    u = np.zeros((im1.shape))
    v = np.zeros((im1.shape))
    
    for _ in range(n_iters):
        
        # Iterative corrections to the displacement estimate
        ua = cv2.filter2D(u, -1, Ld)
        va = cv2.filter2D(v, -1, Ld)

        # Calculate P and D
        P = Ix * ua + Iy * va + It
        D = lmbd + Ixx + Iyy
        
        # Update u and v
        u = ua - Ix * (P / D)
        v = va - Iy * (P / D)
        
    return u, v


def hornschunck_improved(im1, im2, n_iters, lmbd):
    """
    Implementation of Horn-Schunck algorithm.

    Args:
        im1 (matrix): first image matrix (grayscale)
        im2 (matrix): first image matrix (grayscale)
        n_iters (int): number of iterations
        lmbd (float): parameter
    """
    
    # (1) Odvodi:    
    im1x, im1y = gaussderiv(im1, 1)
    im2x, im2y = gaussderiv(im2, 1)
    It = gausssmooth(im2 - im1, 1)
    
    # Pixel-wise average
    Ix = (im1x + im2x) / 2
    Iy = (im1y + im2y) / 2
    
    # (2) Mnozenja:   
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    
    Ld = np.array([[0.0, 0.25, 0.0],
                   [0.25, 0.0, 0.25],
                   [0.0, 0.25, 0.0]])
    
    
    # Pripravimo u in v -- tokrat jih inicializiramo kot output LucasK metode!
    
    u, v = lucaskanade(im1, im2, 3)
    
    for _ in range(n_iters):
        
        # Iterative corrections to the displacement estimate
        ua = cv2.filter2D(u, -1, Ld)
        va = cv2.filter2D(v, -1, Ld)

        # Calculate P and D
        P = Ix * ua + Iy * va + It
        D = lmbd + Ixx + Iyy
        
        # Update u and v
        u = ua - Ix * (P / D)
        v = va - Iy * (P / D)
        
    return u, v



