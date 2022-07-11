import numpy as np 
from auxiliary_funcs import project_cam,project_cam_lookat,system_transform
from calculate_normals import calculate_normals
from ambient_light import ambient_light
from diffuse_light import diffuse_light
from specular_light import specular_light
from shaders import shade_gouraud,shade_phong
from rasterizer import rasterize
def render_object(shader,focal_length,eye,lookat,up,bg_color,M,N,H,W,verts,verts_colors,face_indices,k_a,k_d,k_s,n,light_pos,light_intensities,Ia):
		# Calculate normal vectors of vertices
		Normals = calculate_normals(verts,face_indices)
		
		# Project points 
		P,depth = project_cam_lookat(focal_length,eye,lookat,up,verts)
	
		#Find pixel values
		VertsRast = rasterize(P,M,N,H,W)
		#Create image background black
		Img = np.zeros((N,M,3))

		for i in range(0,2):
			Img[i,:,:] = bg_color[i]


		for p, point in enumerate(verts):
			verts_colors[p] += ambient_light(k_a,Ia)+diffuse_light(point,Normals[p],verts_colors[p],k_d,light_pos,light_intensities,)+specular_light(point,Normals[p],verts_colors[p],eye,k_s,n,light_pos,light_intensities)

		triangle_depths = np.array(np.mean(depth[face_indices], axis = 1))
		reversed_depths =  list(np.flip(np.argsort(triangle_depths)))
		for r in reversed_depths:
			triangle_index =  face_indices[r]
			verts_p = np.array(VertsRast[triangle_index])
			verts_c = np.array(verts_colors[triangle_index])
			verts_n = Normals[triangle_index]
			bcoords =  np.mean(verts[triangle_index,:],axis=0)

		if shader == 1:	
			Img = shade_gouraud(verts_p,verts_n,verts_c,bcoords,eye,k_a,k_d,k_s,n,light_pos,light_intensities,Ia,Img)
		elif shader == 2:
			Img = shade_phong(verts_p,verts_n,verts_c,bcoords,eye,k_a,k_d,k_s,n,light_pos,light_intensities,Ia,Img)
		else:
			print("Wrong shader code input")
			return 1

		return Img
 