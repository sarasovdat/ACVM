import cv2
import numpy as np
from math import floor

from ex1_utils import gausssmooth
from ex2_utils import get_patch, generate_responses_1


###############################################################################################

def generate_my_responses():
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[30, 50] = 0.8
    responses[50, 30] = 0.5
    responses[20, 20] = 2.5
    responses[60, 90] = 0.25
    responses[90, 60] = 0.75
    responses[10, 10] = -0.5
    return gausssmooth(responses, 10)

def create_kernel(kernel_size):
    # Liho število!
    if kernel_size[0] % 2 == 0:
        kernel_size[0] += 1
    if kernel_size[1] % 2 == 0:
        kernel_size[1] += 1
    
    height = kernel_size[0]
    width = kernel_size[1]
    
    kernel_x = np.zeros((height, width))
    kernel_y = np.zeros((height, width))
    
    low_x = - width // 2 + 1    
    for i in range(width):
        kernel_x[:,i] = height * [low_x]
        low_x += 1   
        
        
    low_y = - height // 2 + 1   
    for i in range(height):
        kernel_y[i] = width * [low_y]
        low_y += 1
    
    return kernel_x, kernel_y
    
    
def mean_shift(response, x_start, y_start, kernel_size, epsilon, max_iter = 500):
    """
    Response: generate_responses_1()
    (x_start, y_start): starting point
    kernel_size: size of kernel
    Epsilon: epsilon to check convergence
    Max_iter: to stop if not converge
    """
    
    xi, yi = create_kernel((kernel_size, kernel_size))
    
    x, y = x_start, y_start
    positions = []
    
    for i in range(max_iter):
        wi, inliers = get_patch(response, (x, y), (kernel_size, kernel_size))
        
        x_new_pos = np.sum(xi * wi) / np.sum(wi)
        y_new_pos = np.sum(yi * wi) / np.sum(wi) 
        
        x = x + x_new_pos
        y = y + y_new_pos
        
        positions.append((int(floor(x)), int(floor(y))))
        
        if np.abs(x_new_pos) < epsilon and np.abs(y_new_pos) < epsilon:
            break
    
    return int(floor(x)), int(floor(y)), positions, i


###############################################################################################
###############################################################################################


if __name__ == "__main__":
    responses = generate_my_responses()
    kernel_size = 5
    epsilon = 0.01
    x_start, y_start = 50, 60

    x, y, positions, iters = mean_shift(responses, x_start, y_start, kernel_size, epsilon)
    print(x, y, responses[y][x], iters)
    
    img_layer = 255 * 500 * responses
    img = cv2.merge([img_layer, img_layer, img_layer])
    
    # Starting point
    cv2.circle(img, (x_start, y_start), radius = 0, color = (0, 0, 255), thickness = 2)
    
    # Moving path
    for point in positions:
        cv2.circle(img, point, radius = 0, color = (200, 255, 255), thickness = 1)
        
    # Ending point
    cv2.circle(img, (x, y), radius = 0, color = (0, 255, 0), thickness = 3)
    
    resized_image = cv2.resize(img, (500, 500)) 
    cv2.imwrite("fig_my3.png", resized_image)
