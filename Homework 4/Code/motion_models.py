import numpy as np
import matplotlib.pyplot as plt
from ex4_utils import kalman_step
import sympy as sp
import math

# Assignment 1: Motion models and Kalman filter


def random_walk(q, r):
    """ Prepares input matrices for Kalman step for random walk

    Args:
        q (float): parameter
        r (float): parameter

    Returns:
        Matrices A, C, Q_i, R_i
    """
    Fi = np.array([[1, 0], [0, 1]])
    H = np.array([[1, 0], [0, 1]])
    Q_i = q * np.array([[1, 0], [0, 1]])
    R_i = r * np.array([[1, 0], [0, 1]])
    
    A = Fi
    C = H
    
    return A, C, Q_i, R_i


def nearly_const_velocity(q, r):
    """ Prepares input matrices for Kalman step for nearly constant velocity

    Args:
        q (float): parameter
        r (float): parameter

    Returns:
        Matrices A, C, Q_i, R_i
    """
    Fi = np.array([[1, 0, 1, 0],
                   [0, 1, 0, 1],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    Q_i = q * np.array([[1/3, 0, 1/2, 0],
                        [0, 1/3, 0, 1/2],
                        [1/2, 0, 1, 0],
                        [0, 1/2, 0, 1]])
    R_i = r * np.array([[1, 0], [0, 1]])
    
    A = Fi
    C = H
    
    return A, C, Q_i, R_i

def nearly_const_acceleration(q, r):
    """ Prepares input matrices for Kalman step for nearly constant acceleration

    Args:
        q (float): parameter
        r (float): parameter

    Returns:
        Matrices A, C, Q_i, R_i
    """
    Fi = np.array([[1, 0, 1, 0, 1/2, 0],
                   [0, 1, 0, 1, 0, 1/2],
                   [0, 0, 1, 0, 1, 0],
                   [0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])
    Q_i = q * np.array([[1/20, 0, 1/8, 0, 1/6, 0],
                        [0, 1/20, 0, 1/8, 0, 1/6],
                        [1/8, 0, 1/3, 0, 1/2, 0],
                        [0, 1/8, 0, 1/3, 0, 1/2],
                        [1/6, 0, 1/2, 0, 1, 0],
                        [0, 1/6, 0, 1/2, 0, 1]])   
    R_i = r * np.array([[1, 0], [0, 1]])
    
    A = Fi
    C = H
    
    return A, C, Q_i, R_i    
    
# --------------------------------------------------------------------------------------------------------------- #

def main(name):

    # Example 1
    if name == "Example1":
    
        N = 40
        v = np.linspace(5 * math.pi, 0, N) 
        x = np.cos(v) * v
        y = np.sin(v) * v   
        
        fig, axis = plt.subplots(3, 5)
        fig.set_figwidth(3*5)
        fig.set_figheight(3*3)
        
        parameters = [(100, 1), (5, 1), (1, 1), (1, 5), (1, 100)]
        
        for i in range(3):
            for k in range(len(parameters)):
                q = parameters[k][0]
                r = parameters[k][1]
                
                if i == 0:
                    A, C, Q_i, R_i = random_walk(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("RW: q = " + str(q) + ", r = " + str(r), size = 9)            
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                    
                if i == 1:
                    A, C, Q_i, R_i = nearly_const_velocity(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("NCV: q = " + str(q) + ", r = " + str(r), size = 9)        
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                
                if i == 2:
                    A, C, Q_i, R_i = nearly_const_acceleration(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("NCA: q = " + str(q) + ", r = " + str(r), size = 9)        
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                    
        plt.savefig("Kalman_filter_example_1.png", bbox_inches='tight')
        #plt.show()
    
    # -------------------------------------------------------- #
    
    # Example 2:
    if name == "Example2":
        N = 40
        v = np.linspace(5 * math.pi, 0, N) 
        x = np.cos(v) * v
        y = v

        fig, axis = plt.subplots(3, 5)
        fig.set_figwidth(3*5)
        fig.set_figheight(3*3)
        
        parameters = [(100, 1), (5, 1), (1, 1), (1, 5), (1, 100)]
        
        for i in range(3):
            for k in range(len(parameters)):
                q = parameters[k][0]
                r = parameters[k][1]
                
                if i == 0:
                    A, C, Q_i, R_i = random_walk(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("RW: q = " + str(q) + ", r = " + str(r), size = 9)            
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                    
                if i == 1:
                    A, C, Q_i, R_i = nearly_const_velocity(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("NCV: q = " + str(q) + ", r = " + str(r), size = 9)        
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                
                if i == 2:
                    A, C, Q_i, R_i = nearly_const_acceleration(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("NCA: q = " + str(q) + ", r = " + str(r), size = 9)        
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                    
        plt.savefig("Kalman_filter_example_2.png", bbox_inches='tight')
        #plt.show()
        
    # -------------------------------------------------------- #
    
    # Example 3:
    if name == "Example3":
        x = np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2])
        y = np.array([1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1])

        fig, axis = plt.subplots(3, 5)
        fig.set_figwidth(3*5)
        fig.set_figheight(3*3)
        
        parameters = [(100, 1), (5, 1), (1, 1), (1, 5), (1, 100)]
        
        for i in range(3):
            for k in range(len(parameters)):
                q = parameters[k][0]
                r = parameters[k][1]
                
                if i == 0:
                    A, C, Q_i, R_i = random_walk(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("RW: q = " + str(q) + ", r = " + str(r), size = 9)            
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                    axis[i, k].set_xlim([0, 8])
                    axis[i, k].set_ylim([0, 8])
                    
                if i == 1:
                    A, C, Q_i, R_i = nearly_const_velocity(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("NCV: q = " + str(q) + ", r = " + str(r), size = 9)        
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                    axis[i, k].set_xlim([0, 8])
                    axis[i, k].set_ylim([0, 8])
                    
                if i == 2:
                    A, C, Q_i, R_i = nearly_const_acceleration(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("NCA: q = " + str(q) + ", r = " + str(r), size = 9)        
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                    axis[i, k].set_xlim([0, 8])
                    axis[i, k].set_ylim([0, 8])
                    
        plt.savefig("Kalman_filter_example_3.png", bbox_inches='tight')
        #plt.show()
    
    # -------------------------------------------------------- #
    
    # Example 4:
    if name == "Example4":
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 10, 1, 10, 1, 10, 1, 10, 1, 10])

        fig, axis = plt.subplots(3, 5)
        fig.set_figwidth(3*5)
        fig.set_figheight(3*3)
        
        parameters = [(100, 1), (5, 1), (1, 1), (1, 5), (1, 100)]
        
        for i in range(3):
            for k in range(len(parameters)):
                q = parameters[k][0]
                r = parameters[k][1]
                
                if i == 0:
                    A, C, Q_i, R_i = random_walk(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("RW: q = " + str(q) + ", r = " + str(r), size = 9)            
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                    
                if i == 1:
                    A, C, Q_i, R_i = nearly_const_velocity(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("NCV: q = " + str(q) + ", r = " + str(r), size = 9)        
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                    
                if i == 2:
                    A, C, Q_i, R_i = nearly_const_acceleration(q, r)
                    
                    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
                    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

                    sx[0] = x[0] 
                    sy[0] = y[0]
                    
                    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
                    state[0] = x[0] 
                    state[1] = y[0]
                    covariance = np.eye(A.shape[0] , dtype=np.float32)
                    
                    for j in range(1, x.size):
                        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                        sx[j] = state[0] 
                        sy[j] = state[1]
                    
                    axis[i, k].plot(x, y, color = "coral", marker = '.')
                    axis[i, k].plot(sx, sy, "cornflowerblue", marker = '.')
                    axis[i, k].set_title("NCA: q = " + str(q) + ", r = " + str(r), size = 9)        
                    axis[i, k].tick_params(axis='both', which='major', labelsize=5)
                    
        plt.savefig("Kalman_filter_example_4.png", bbox_inches='tight')
        #plt.show()
    
# ------------------------------------------------------------------------------------------------------------------------ #

# ODKOMENTIRAJ ZA IZRIS EXAMPLE GRAFOV:

#main("Example1")  #reference figure -- iz predavanj
#main("Example2")
#main("Example3")
#main("Example4")