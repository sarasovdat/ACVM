from hashlib import shake_128
import cv2
import numpy as np
from math import floor

from ex2_utils import get_patch, generate_responses_1, Tracker, create_epanechnik_kernel, extract_histogram, backproject_histogram
from mean_shift_mode_seeking import create_kernel


###############################################################################################    

class MSParams:
    def __init__(self):
        self.sigma = 1
        self.n_bins = 16
        self.max_iter = 20
        self.eps = 0.05
        self.epsilon = 0.01
        self.patch_factor = 1.05
        self.alpha = 0.01
        self.enlarge_factor = 2
        
        
class MeanShiftTracker(Tracker):
    
    def initialize(self, image, region):
                    
        # FRAME 1 : INITIALIZATION 
        
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.region = region   
             
        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)
        
        s1 = round(self.region[2])
        s2 = round(self.region[3])       
        if s1 % 2 == 0:
            s1 += 1  
        if s2 % 2 == 0:
            s2 += 1            
        self.size = (s1, s2)         
            
        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (np.round(self.region[0] + self.region[2] / 2), 
                         np.round(self.region[1] + self.region[3] / 2))  
        

        # EXTRACT HISTOGRAM q USING KERNEL (Epanechnikov)
        
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)  
        
        patch, inliers = get_patch(image, self.position, self.size)
        self.q = extract_histogram(patch, self.parameters.n_bins, self.kernel)
        
    
        
    def track(self, image):
        
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            self.region = [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]
            return self.region
        
        x, y = self.position
        xi, yi = create_kernel((self.size[1], self.size[0]))
        
        
        for i in range(self.parameters.max_iter):
            
            patch, inliers = get_patch(image, (x, y), self.size)   
            
            # Extract p using kernel       
            p = extract_histogram(patch, self.parameters.n_bins, self.kernel)
            # Calculate weights
            v = np.sqrt(self.q / (p + self.parameters.eps))
            # Backproject within extracted patch using weights v
            wi = backproject_histogram(patch, v, self.parameters.n_bins) 
            
            x_new_pos = np.sum(xi * wi) / np.sum(wi)
            y_new_pos = np.sum(yi * wi) / np.sum(wi)  
            
            x = x + x_new_pos
            y = y + y_new_pos
            
            if np.abs(x_new_pos) < self.parameters.epsilon and np.abs(y_new_pos) < self.parameters.epsilon:
                break
 
        # Update model     
        patch, inliers = get_patch(image, (x, y), self.size) 
        q_tilda = extract_histogram(patch, self.parameters.n_bins, self.kernel)
        
        q_new = (1 - self.parameters.alpha) * self.q + self.parameters.alpha * q_tilda
        self.q = q_new
            
        self.region = [x - self.size[0] / 2, y - self.size[1] / 2, self.size[0], self.size[1]]

        return self.region