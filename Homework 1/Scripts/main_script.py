import numpy as np
import matplotlib . pyplot as plt
import cv2
from ex1_utils import rotate_image , show_flow 
from ex1_code import lucaskanade, hornschunck, hornschunck_improved
import time

# Test both methods on other images (at least three more pairs of images)

# DODAJ ŠE IMENA ZA FAJLE V FUNKCIJE!!!!!

def test_method(im1, im2, method, name):
    """
    Function for testing the method on images im1 and im2. 
    Answer to the task
            "
              Test both methods on other images (at least 3 more pairs of images), include results in the report and comment them. 
              You can use images included in the project material or (even better) add your own examples.
            "

    Args:
        im1 (matrix): first image matrix (grayscale)
        im2 (matrix): second image matrix (grayscale)
        method (str): method to be tested ("LucasK" / "HornS")
        name (str): name for figure when saving
    """
    
    im1 = cv2.imread(im1, cv2.COLOR_BGR2GRAY).astype(np.float32) 
    im2 = cv2.imread(im2, cv2.COLOR_BGR2GRAY).astype(np.float32) 
    
    # Normalization
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    
    if method == "LucasK":
        U_lk , V_lk = lucaskanade(im1 , im2 , 3)
        fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
        ax1_11.imshow(im1) 
        ax1_12.imshow(im2)
        show_flow(U_lk , V_lk , ax1_21 , type = 'angle')
        show_flow(U_lk , V_lk , ax1_22 , type = 'field', set_aspect = True)
        fig1.suptitle ('Lucas-Kanade Optical Flow')
        
        plt.tight_layout()
        plt.savefig('Figures/LucasK_method_' + str(name) + '.png', bbox_inches = 'tight')
        #plt.show()

    if method == "HornS":
        U_hs , V_hs = hornschunck(im1 , im2 , 1000, 0.5)
        fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2) 
        ax2_11.imshow(im1)
        ax2_12.imshow(im2)
        show_flow(U_hs , V_hs , ax2_21 , type = 'angle')
        show_flow(U_hs , V_hs , ax2_22 , type = 'field' , set_aspect = True) 
        fig2.suptitle('Horn-Schunck Optical Flow')
        
        plt.tight_layout()
        plt.savefig('Figures/HornS_method_' + str(name) + '.png', bbox_inches = 'tight')
        #plt.show()
        

def parameters_impact(im1, im2, method, name):
    """
    Function for testing parameters impact on the method. 
    Answer to the task 
            " 
              Which parameters need to be determined for Lucas-Kanade and Horn-Schunck? 
              How do these parameters impact on optical-flow performance? 
              Show examples and discuss your observations which parameters are optimal in which cases.
            "
            
    Args:
        im1 (matrix): first image matrix (grayscale)
        im2 (matrix): second image matrix (grayscale)
        method (str): method to be tested ("LucasK" / "HornS")
        name (str): name for figure when saving
    """
    
    im1 = cv2.imread(im1, cv2.COLOR_BGR2GRAY).astype(np.float32) 
    im2 = cv2.imread(im2, cv2.COLOR_BGR2GRAY).astype(np.float32) 
    
    # Normalization
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    
    if method == "LucasK":
        U_lk1, V_lk1 = lucaskanade(im1 , im2 , 3)
        U_lk2, V_lk2 = lucaskanade(im1 , im2 , 5)
        U_lk3, V_lk3 = lucaskanade(im1 , im2 , 10)
        U_lk4, V_lk4 = lucaskanade(im1 , im2 , 15)
        U_lk5, V_lk5 = lucaskanade(im1 , im2 , 20)
        U_lk6, V_lk6 = lucaskanade(im1 , im2 , 25)
        
        fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)   
        
        show_flow(U_lk1, V_lk1, ax1, type = 'field', set_aspect = True)   
        show_flow(U_lk2, V_lk2, ax2, type = 'field', set_aspect = True)      
        show_flow(U_lk3, V_lk3, ax3, type = 'field', set_aspect = True)   
        show_flow(U_lk4, V_lk4, ax4, type = 'field', set_aspect = True)
        show_flow(U_lk5, V_lk5, ax5, type = 'field', set_aspect = True)   
        show_flow(U_lk6, V_lk6, ax6, type = 'field', set_aspect = True)
        
        ax1.title.set_text("N = 3")
        ax2.title.set_text("N = 5")
        ax3.title.set_text("N = 10")
        ax4.title.set_text("N = 15")
        ax5.title.set_text("N = 20")
        ax6.title.set_text("N = 25")
        
        fig1.suptitle('Lucas-Kanade: parameters impact')
        
        plt.tight_layout()
        plt.savefig('Figures/LucasK_parameters_' + str(name) + '.png', bbox_inches = 'tight')
        #plt.show()
        

    if method == "HornS":
        
        # Calibrating the lambda paramater on fixed (1000) number of iterations
        U_hs1, V_hs1 = hornschunck(im1 , im2 , 1000, 0.1)
        U_hs2, V_hs2 = hornschunck(im1 , im2 , 1000, 0.5)
        U_hs3, V_hs3 = hornschunck(im1 , im2 , 1000, 2)
        
        # Calibrating the number of iterations parameter on fixed lambda (0.5)
        U_hs4, V_hs4 = hornschunck(im1 , im2 , 100, 0.5)
        U_hs5, V_hs5 = hornschunck(im1 , im2 , 500, 0.5)
        U_hs6, V_hs6 = hornschunck(im1 , im2 , 1000, 0.5)

        fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)  
        
        show_flow(U_hs1, V_hs1, ax1, type = 'field', set_aspect = True)   
        show_flow(U_hs2, V_hs2, ax2, type = 'field', set_aspect = True)      
        show_flow(U_hs3, V_hs3, ax3, type = 'field', set_aspect = True)   
        show_flow(U_hs4, V_hs4, ax4, type = 'field', set_aspect = True)
        show_flow(U_hs5, V_hs5, ax5, type = 'field', set_aspect = True)   
        show_flow(U_hs6, V_hs6, ax6, type = 'field', set_aspect = True)
        
        ax1.title.set_text("n = 1000, lmbd = 0.1")
        ax2.title.set_text("n = 1000, lmbd = 0.5")
        ax3.title.set_text("n = 1000, lmbd = 2")
        ax4.title.set_text("n = 100, lmbd = 0.5")
        ax5.title.set_text("n = 500, lmbd = 0.5")
        ax6.title.set_text("n = 1000, lmbd = 0.5")
        
        fig1.suptitle('Horn-Schunck: parameters impact')
        
        plt.tight_layout()
        plt.savefig('Figures/HornS_parameters_' + str(name) + '.png', bbox_inches = 'tight')
        #plt.show()
        
   
def measure_time(im1, im2, method):
    """
    Function for measuring time for given method.
    Answer to the task 
            " 
              Measure time for Lucas-Kanade and Horn-Schunck optical flow methods and report measurements. 
              Can you speed-up Horn-Schunck method, e.g., by initializing it with output of Lucas-Kanade? 
              What is the speed and performance of the improved Horn-Schunck?
            "
            
     Args:
        im1 (matrix): first image matrix (grayscale)
        im2 (matrix): second image matrix (grayscale)
        method (str): method to be tested ("LucasK" / "HornS" / "Improved HornS")
    """
     
    im1 = cv2.imread(im1, cv2.COLOR_BGR2GRAY).astype(np.float32) 
    im2 = cv2.imread(im2, cv2.COLOR_BGR2GRAY).astype(np.float32) 
    
    # Normalization
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    
    if method == "LucasK":
        start = time.time()
        lucaskanade(im1 , im2 , 3)
        stop = time.time()
        total_time = stop - start
        print("LucasK time : " + str(total_time))
    
    if method == "HornS":
        start = time.time()
        hornschunck(im1 , im2 , 1000, 0.5)
        stop = time.time()
        total_time = stop - start
        print("HornS time : " + str(total_time))
    
    if method == "Improved HornS":
        start = time.time()
        hornschunck_improved(im1 , im2 , 1000, 0.5)
        stop = time.time()
        total_time = stop - start
        print("Improved HornS time : " + str(total_time))
     
    return None
     
    
##########################################################################################################################################
    
if __name__ == "__main__":
    

    #################################################################################################
    # TESTING METHODS
    
    """
    # LucasK: 
    test_method("./disparity/cporta_left.png", "./disparity/cporta_right.png", "LucasK", "cporta")
    test_method("./disparity/office_left.png", "./disparity/office_right.png", "LucasK", "office")
    test_method("./disparity/office2_left.png", "./disparity/office2_right.png", "LucasK", "office2")
    
    # HornS:
    test_method("./disparity/cporta_left.png", "./disparity/cporta_right.png", "HornS", "cporta")
    test_method("./disparity/office_left.png", "./disparity/office_right.png", "HornS", "office")
    test_method("./disparity/office2_left.png", "./disparity/office2_right.png", "HornS", "office2")
    
    
    #################################################################################################
    # PARAMETER IMPACT
    
    # LucasK:
    parameters_impact("./disparity/cporta_left.png", "./disparity/cporta_right.png", "LucasK", "cporta")
    parameters_impact("./disparity/office_left.png", "./disparity/office_right.png", "LucasK", "office")
    parameters_impact("./disparity/office2_left.png", "./disparity/office2_right.png", "LucasK", "office2")
    
    # HornS:
    parameters_impact("./disparity/cporta_left.png", "./disparity/cporta_right.png", "HornS", "cporta")
    parameters_impact("./disparity/office_left.png", "./disparity/office_right.png", "HornS", "office")
    parameters_impact("./disparity/office2_left.png", "./disparity/office2_right.png", "HornS", "office2")
    

    #################################################################################################
    # TIME MEASURE
    
    # LucasK:
    measure_time("./disparity/cporta_left.png", "./disparity/cporta_right.png", "LucasK")
    measure_time("./disparity/office_left.png", "./disparity/office_right.png", "LucasK")
    measure_time("./disparity/office2_left.png", "./disparity/office2_right.png", "LucasK")
    
    # HornS:
    measure_time("./disparity/cporta_left.png", "./disparity/cporta_right.png", "HornS")
    measure_time("./disparity/office_left.png", "./disparity/office_right.png", "HornS")
    measure_time("./disparity/office2_left.png", "./disparity/office2_right.png", "HornS")
    
    # Improved HornS:
    measure_time("./disparity/cporta_left.png", "./disparity/cporta_right.png", "Improved HornS")
    measure_time("./disparity/office_left.png", "./disparity/office_right.png", "Improved HornS")
    measure_time("./disparity/office2_left.png", "./disparity/office2_right.png", "Improved HornS")
    """