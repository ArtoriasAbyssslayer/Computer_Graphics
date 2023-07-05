from src.common_utils.load_vertices import load_IlluminationData_npy as load_data_npy
from src.common_utils.render import render_object
from src.common_utils.phongMaterial import PhongMaterial as phongMaterial
from src.common_utils.pointLight import PointLight as pointLight
import cv2 as cv

import time 
import numpy as np 



def main(*args):
    verts, vcolors, face_indices, eye, up, lookat, ka, kd, ks, n_phong, light_positions,\
    light_intensities, M, N, W, H, light_amb,bg_color,focal = load_data_npy(filename='./GHW3-assets/h3.npy')
    
    mat = phongMaterial(ka, kd, ks, n_phong)
    lights = pointLight(phongMaterial,light_positions, light_intensities)
    tic = time.time()
    bg_color = np.zeros((M,N,3))
    shader = args[0]
    Y = render_object(shader,focal,eye,lookat,up,bg_color,M,N,H,W,verts,vcolors,face_indices,mat,lights,light_amb)
    cv.imshow(Y)
    toc = time.perf_counter()
    print(f"Shading finished in {(toc - tic)/60:0.0f} minutes {(toc - tic)%60:0.0f} seconds")
    
    
if __name__ == "__main__":
    shader = 1
    main(shader)