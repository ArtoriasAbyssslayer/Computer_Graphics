from src.common_utils.load_vertices import load_IlluminationData_npy as load_data_npy
from src.common_utils.render import render_object,renderImageToFile
from src.common_utils.phongMaterial import PhongMaterial as phongMaterial
from src.common_utils.pointLight import PointLight as pointLight
import time 
import numpy as np 



def main(*args):
    verts, vcolors, face_indices, eye, up, lookat, ka, kd, ks, n_phong, light_positions,\
    light_intensities, M, N, W, H, light_amb,bg_color,focal = load_data_npy(filename='./GHW3-assets/h3.npy')
    # Cases of produced images
    # Case 0: Only Ambient Light
    # Case 1: Only Diffuse Reflection 
    # Case 2: Only Specular Reflection 
    # Case 4: Full Reflection Model with N phong parameter 
    mat = phongMaterial(ka, kd, ks, n_phong)
    lights = pointLight(phongMaterial,light_positions, light_intensities)
    LightingParam = [0,1,2,3]
    
    bg_color = np.ones((M,N,3))
    shader = args[0]
    for param in LightingParam:
        if param == 0:
            tic = time.time()
            Y = render_object(shader,focal,eye,lookat,up,bg_color,M,N,H,W,verts,vcolors,face_indices,mat,lights,light_amb,param)
            renderImageToFile(Y , 'Gouraud_Ambient',save=True)
            toc = time.time()
            min = int((toc - tic)/60)
            print(f"Shading Ambient Only Lighting finished in {min} minutes {(toc - tic)%60:0.0f} seconds")
        elif param == 1:
            tic = time.time()
            Y = render_object(shader,focal,eye,lookat,up,bg_color,M,N,H,W,verts,vcolors,face_indices,mat,lights,light_amb,param)
            renderImageToFile(Y , 'Gouraud_Diffuse',save=True)
            toc = time.time()
            min = int((toc - tic)/60)
            print(f"Shading Diffuse Only Lighting finished in {min} minutes {(toc - tic)%60:0.0f} seconds")
        elif param == 0:
            tic = time.time()
            Y = render_object(shader,focal,eye,lookat,up,bg_color,M,N,H,W,verts,vcolors,face_indices,mat,lights,light_amb,param)
            renderImageToFile(Y,'Gouraud_Specular',save=True)
            toc = time.time()
            min = int((toc - tic)/60)
            print(f"Shading Specular Only Lighting  finished in {min} minutes {(toc - tic)%60:0.0f} seconds")
        else:
            tic = time.time()
            Y = render_object(shader,focal,eye,lookat,up,bg_color,M,N,H,W,verts,vcolors,face_indices,mat,lights,light_amb,param)
            renderImageToFile(Y,'Gouraud_Full_Phong_Reflection',save=True)
            toc = time.time()
            min = int((toc - tic)/60)
            print(f"Shading Phong Full Lighting finished in {min} minutes {(toc - tic)%60:0.0f} seconds")
    
if __name__ == "__main__":
    shader = 0
    main(shader)