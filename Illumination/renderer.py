import numpy as np 
from auxiliary_funcs import project_cam,project_cam_lookat,system_transform
from calculate_normals import calculate_normals
from ambient_light import ambient_light
from shaders import shade_gouraud
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

		# Sort the order of every triangle
		triangles_ordered = np.sort(depth,axis=0)
		triangles_ordered = triangles_ordered[::-1] 
		triangles_ordered = np.mean(triangles_ordered)
		# Find depth of each triangle
		for i in range(0,face_indices.shape[1]):
			index = face_indices[:,int(triangles_ordered[i])]
			# verticies position values
			verts_p = VertsRast[:,index]
			# verticies normals
			verts_n = Normals[:,index]
			# verticies colors
			verts_c = verts_colors[:,index]
			# calculate gravitational centers
			bcoords = np.mean(verts[:,idx],axis=1)

		if shader == 1:
			Img = shade_gouraud(verts_p,verts_n,verts_c,bcoords,cam_pos,k_a,k_d,k_s,n,light_pos,light_intesities,I_a,Img)
		elif shader == 2:
			Img = shade_phong(verts_p,verts_n,verts_c,bcoords,cam_pos,k_a,k_d,k_s,n,light_pos,light_intesities,I_a,Img)
		else:
			print("Wrong shader code input")
			return 1

		return Img
 