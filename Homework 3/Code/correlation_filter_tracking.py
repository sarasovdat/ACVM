import numpy as np
import cv2

from ex2_utils import get_patch
from ex3_utils import create_cosine_window, create_gauss_peak

from utils import tracker
import time

# CORRELATION FILTER TRACKING

# 1) Create workspace
# python create_workspace.py --workspace_path '/Users/sarabizjak/Documents/IŠRM magistrski/Advanced Computer Vision Methods/ACVM/Homework 3/workspace-dir' --dataset vot2014
# 2) Tracker integration and running
# python evaluate_tracker.py --workspace_path '/Users/sarabizjak/Documents/IŠRM magistrski/Advanced Computer Vision Methods/ACVM/Homework 3/workspace-dir' --tracker cft_tracker

# 3) Results visualization and tracking performance evaluation
# VISUALIZATION
# python visualize_result.py --workspace_path '/Users/sarabizjak/Documents/IŠRM magistrski/Advanced Computer Vision Methods/ACVM/Homework 3/workspace-dir' --tracker cft_tracker --sequence bolt
# PERFORMANCE
# python compare_trackers.py --workspace_path '/Users/sarabizjak/Documents/IŠRM magistrski/Advanced Computer Vision Methods/ACVM/Homework 3/workspace-dir' --trackers cft_tracker --sensitivity 100

# 4)
#

"""
cft_tracker:  # use tracker identifier as you want to call your tracker from terminal
  tracker_path: /Users/sarabizjak/Documents/IŠRM magistrski/Advanced Computer Vision Methods/ACVM/Homework 3/correlation_filter_tracking.py # path to the Python script with your tracker
  class_name: CFTTracker  # class name of your tracker
  paths:  # optionally: additional Python paths you want to include
"""
 
class CFTParams:
    def __init__(self):
        self.alpha = 0.1
        self.sigma = 2
        self.enlarge_factor = 1.2
        self.lam = 0.00001
        
        
class CFTTracker(tracker.Tracker):
    
    def __init__(self):
        self.parameters = CFTParams()
        
    def name(self):
        return 'CFTTracker'
    
    def initialize(self, image, region):
        
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.region = region          
        
        s1 = round(self.region[2] * self.parameters.enlarge_factor)
        s2 = round(self.region[3] * self.parameters.enlarge_factor) 
        if s1 % 2 == 0:
            s1 += 1  
        if s2 % 2 == 0:
            s2 += 1            
        self.size = (s1, s2)    
        
        self.position = (np.round(self.region[0] + self.region[2] / 2), 
                         np.round(self.region[1] + self.region[3] / 2))  
        
        # Localization step: filter application
        self.G = create_gauss_peak((s1, s2), self.parameters.sigma)
        self.G_fft = np.fft.fft2(self.G)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          
        patch, inliers = get_patch(image, self.position, (self.size[0], self.size[1]))       
        cosine_window = create_cosine_window((self.size[0], self.size[1]))
        self.F = patch * cosine_window
        self.F_fft = np.fft.fft2(self.F)
        self.F_fft_conj = np.conj(self.F_fft)
        
        self.filter = (self.G_fft * self.F_fft_conj) / (self.F_fft * self.F_fft_conj + self.parameters.lam)
        
        
    def track(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Update step: filter learning      
        patch, inliers = get_patch(image, self.position, (self.size[0], self.size[1]))
        cosine_window = create_cosine_window((self.size[0], self.size[1]))
        F = patch * cosine_window
        F_fft = np.fft.fft2(F)
        
        response = self.filter * F_fft
        R = np.fft.ifft2(response)
        
        y, x = np.unravel_index(np.argmax(R), R.shape)
        
        if x > (self.size[0] / 2):
            x = x - self.size[0]
        if y > (self.size[1] / 2):
            y = y - self.size[1]
            
        self.position = (self.position[0] + x, self.position[1] + y)
        
        # New filter
        patch_new, inliers_new = get_patch(image, self.position, (self.size[0], self.size[1])) 
        cosine_window_new = create_cosine_window((self.size[0], self.size[1]))
        F_new = patch_new * cosine_window_new
        F_fft_new = np.fft.fft2(F_new)
        F_fft_conj_new = np.conj(F_fft_new)
        
        filter_new = (self.G_fft * F_fft_conj_new) / (F_fft_new * F_fft_conj_new + 0.000001)
        filter_updated = (1 - self.parameters.alpha) * self.filter + self.parameters.alpha * filter_new      
        self.filter = filter_updated       
        
        self.region = [self.position[0] - self.size[0] / 2, self.position[1] - self.size[1] / 2, self.size[0], self.size[1]]        
            
        return self.region
                
    
    
        
