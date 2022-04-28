import numpy as np
import matplotlib.pyplot as plt
from ex4_utils import kalman_step
import sympy as sp
import math
from scipy import linalg

from ex2_utils import create_epanechnik_kernel, extract_histogram, get_patch
from ex4_utils import sample_gauss
from motion_models import nearly_const_velocity, random_walk, nearly_const_acceleration
from utils import tracker

# 

# 1) Create workspace
# python create_workspace.py --workspace_path '/Users/sarabizjak/Documents/IŠRM magistrski/Advanced Computer Vision Methods/ACVM/Homework 4/workspace-dir' --dataset vot2014
# 2) Tracker integration and running
# python evaluate_tracker.py --workspace_path '/Users/sarabizjak/Documents/IŠRM magistrski/Advanced Computer Vision Methods/ACVM/Homework 4/workspace-dir' --tracker pf_tracker

# 3) Results visualization and tracking performance evaluation
# VISUALIZATION
# python visualize_result.py --workspace_path '/Users/sarabizjak/Documents/IŠRM magistrski/Advanced Computer Vision Methods/ACVM/Homework 4/workspace-dir' --tracker pf_tracker --sequence bolt
# PERFORMANCE
# python compare_trackers.py --workspace_path '/Users/sarabizjak/Documents/IŠRM magistrski/Advanced Computer Vision Methods/ACVM/Homework 4/workspace-dir' --trackers pf_tracker --sensitivity 100

# 4)
#


# Assignment 2: Particle filters


def hellinger(p, q):
    return linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)


class PFParams:
    def __init__(self):
        self.sigma = 10
        self.n_bins = 8
        self.max_iter = 20
        self.eps = 0.05
        self.epsilon = 0.01
        self.patch_factor = 1.05
        self.alpha = 0.05
        self.enlarge_factor = 2
        self.num_particles = 100
        self.prob_sigma = 2
        
        
class PFTracker(tracker.Tracker):
    
    def __init__(self):
        self.parameters = PFParams()
        
    def name(self):
        return "PFTracker"
    
    def initialize(self, image, region):
        
        # INITIALIZATION 
        
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.region = region   
        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        
        s1 = round(self.region[2])
        s2 = round(self.region[3])       
        if s1 % 2 == 0:
            s1 += 1  
        if s2 % 2 == 0:
            s2 += 1            
        self.size = (s1, s2) 
        
        self.position = (np.round(self.region[0] + self.region[2] / 2), 
                         np.round(self.region[1] + self.region[3] / 2))  
        
        # TRACKING MODEL
        
        self.Fi, self.H, self.Q_i, self.R_i = nearly_const_velocity(100, 1) #NCV
        #self.Fi, self.H, self.Q_i, self.R_i = random_walk(100, 1) #RW
        #self.Fi, self.H, self.Q_i, self.R_i = nearly_const_acceleration(100, 1) #NCA
        
        # Particles
        self.state = np.array([self.position[0], self.position[1], 0, 0]) #NCV
        #self.state = np.array([self.position[0], self.position[1]]) #RW
        #self.state = np.array([self.position[0], self.position[1], 0, 0, 0, 0]) #NCA
        
        self.particles = sample_gauss(self.state, self.Q_i, self.parameters.num_particles)
        self.weights = np.ones(self.particles.shape[0])
        
        # Kernel
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)  
        patch, inliers = get_patch(image, self.position, self.size)
        self.q = extract_histogram(patch, self.parameters.n_bins, self.kernel)
        
    def track(self, image):
        
        # Sampling
        weights_norm = self.weights / np.sum(self.weights)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.parameters.num_particles, 1)
        sampled_idxs = np.digitize(rand_samples , weights_cumsumed)
        particles_new = self.particles[sampled_idxs.flatten() , :]
        
        # Adding noise
        #noise = sample_gauss(np.zeros(4), self.Q_i, self.parameters.num_particles)    
        noise = sample_gauss(np.zeros_like(self.state), self.Q_i, self.parameters.num_particles)  
        particles_new = particles_new + noise
        
        # Dynamic model     
        weights_new = np.zeros(self.parameters.num_particles)
        for i, part in enumerate(particles_new):       
            x, y, _, _ = part #NCV
            #x, y = part #RW
            #x, y, _, _, _, _ = part #NCA
            patch, inliers = get_patch(image, (x, y), self.size)
            hist = extract_histogram(patch, self.parameters.n_bins, self.kernel)
            distance = hellinger(hist, self.q)
            dist_to_prob = np.exp(- 1/2 * ((distance ** 2) / (1 * self.parameters.prob_sigma)))
            weights_new[i] = dist_to_prob
            
        # Normalizing weights
        if np.sum(weights_new) != 0:
            weights_new = weights_new / np.sum(weights_new)
        
        # Computing new positions
        x_new_pos = np.sum([p[0] * weights_new[idx] for idx, p in enumerate(particles_new)]) 
        y_new_pos = np.sum([p[1] * weights_new[idx] for idx, p in enumerate(particles_new)]) 
        
        
        # Updating model
        patch, inliers = get_patch(image, (x_new_pos, y_new_pos), self.size)
        q_tilda = extract_histogram(patch, self.parameters.n_bins, self.kernel)
        q_new = (1 - self.parameters.alpha) * self.q + self.parameters.alpha * q_tilda
        self.q = q_new
        
        self.region = self.region = [x_new_pos - self.size[0] / 2, y_new_pos - self.size[1] / 2, self.size[0], self.size[1]]
        
        return self.region