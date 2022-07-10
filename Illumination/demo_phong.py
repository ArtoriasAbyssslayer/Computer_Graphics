from auxiliary_funcs import load_data_npy
from renderer import render_object
import cv2 as cv
import time
import numpy as np
# Phong shader
shader = 2
focal_length = 70

verts, verts_colors, face_indices, eye, up, lookat, ka, kd, ks, n_phong,\
light_pos, light_intensities, M, N, W, H, Ia = load_data_npy(filename='./h3.npy')
# Start measuring 
time
tic = time.perf_counter()
bg_color = np.zeros((N,M,3))
Y = render_object(shader,focal_length,eye,lookat,up,bg_color,M,N,H,W,verts,verts_colors,face_indices,ka,kd,ks,N,light_pos,light_intensities,Ia)
cv.imshow(Y)
toc = time.perf_counter()
print(f"Shading finished in {(toc - tic)/60:0.0f} minutes {(toc - tic)%60:0.0f} seconds")
