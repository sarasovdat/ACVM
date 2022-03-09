import numpy as np
import matplotlib . pyplot as plt
from ex1_utils import rotate_image , show_flow 
from hw1_code import lucaskanade, hornschunck

im1 = np.random.rand(200, 200).astype(np.float32) 
im2 = im1.copy ()
im2 = rotate_image(im2 , -1)
U_lk , V_lk = lucaskanade(im1 , im2 , 3)
U_hs , V_hs = hornschunck(im1 , im2 , 1000, 0.5)

# PLOT 1 : Lucas-Kanade Optical Flow
fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
ax1_11.imshow(im1) 
ax1_12.imshow(im2)
show_flow(U_lk , V_lk , ax1_21 , type = 'angle')
show_flow(U_lk , V_lk , ax1_22 , type = 'field', set_aspect = True)
fig1.suptitle ('Lucas-Kanade Optical Flow')

plt.tight_layout()
plt.savefig('Figures/LucasK_test.png', bbox_inches = 'tight')
        

# PLOT 2 : Horn-Schunck Optical Flow
fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2) 
ax2_11.imshow(im1)
ax2_12.imshow(im2)
show_flow(U_hs , V_hs , ax2_21 , type = 'angle')
show_flow(U_hs , V_hs , ax2_22 , type = 'field' , set_aspect = True) 
fig2.suptitle('Horn-Schunck Optical Flow')

plt.tight_layout()
plt.savefig('Figures/HornS_test.png', bbox_inches = 'tight')


