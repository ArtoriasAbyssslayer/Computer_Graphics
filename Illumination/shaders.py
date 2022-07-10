import numpy as np 
from auxiliary_funcs import *
from scanline_util import TriangleFillingFunction 
from ambient_light import ambient_light as ambient
from diffuse_light import diffuse_light as diffuse
from specular_light import specular_light as specular
import math
def shade_gouraud(verts_p , verts_n, verts_c , bcoords, cam_pos, ka , kd, ks , n, light_positions, light_intensities, Ia , X):
	# Initialize Colors
	vertex_colors = np.zeros((3,3))
	for i in range(0,2):

		# Extract Normal and color
		color = verts_c[:,i]
		N = verts_n[:,i]

		# Add ambient light

		I_ambient = ambient(ka,Ia)

		# Add diffuse light based on bcoords
		I_diffuse = diffuse(bcoords,N,color,kd,light_positions,light_intensities)

		# Add specular_light
		I_specular = specular(bcoords,N,color,cam_pos, ks, n, light_positions, light_intensities)

		verts_colors[i,:] = colors + I_ambient + I_diffuse + I_specular
	tff = TriangleFillingFunction()
	Img = tff.shade_triangle(self,X,verts_p,verts_colors,'gouraud')
	return Img

def shade_phong(verts_p , verts_n, verts_c , bcoords, cam_pos, ka , kd, ks , n, light_positions, light_intensities, Ia , X):
	#Initialize colors
	vertex_colors = np.zeros((3,3))

	vertex_colors = verts_c.transpose()

	# Initialize and find the vertices of the edges in each triangle
	# Implement changed scanline algorithm too
	edges =  np.zeros((2,3,3))
	edges[:,:,0] = [[verts_p[:,0].transpose,1],[verts_p[:,0].transpose,2]]
	edges[:,:,1] = [[verts_p[:,1].transpose,1],[verts_p[:,1].transpose,3]]
	edges[:,:,2] = [[verts_p[:,2].transpose,2],[verts_p[:,2].transpose,3]]


	y_kmin = np.zeros((1,3))
	y_kmax = np.zeros((1,3))

	x_kmin = np.zeros((1,3))
	x_kmax = np.zeros((1,3))


	# Initialize inclination coeff of 3 edges

	m = np.zeros((1,3))

	for k in range(3):
		x = edges[:,0,k]
		y = edges[:,1,k]

		y_kmin[k] = min(y)
		y_kmax[k] = max(y)

		x_kmin[k] = min(x) 
		x_kmax[k] = max(x)

		if x[0] == x[1]:
			m[k] = float('inf')
		else:
			m[k] = (y[1]-y[0])/(x[1]-x[0])


		ymin = min(y_kmin)
		ymax = max(y_kmax)

		# Search indices of the edges intersected by scanline

		indices = np.where(edges[:,1,:]==y_min)
		indices = indices.transpose()
		current_edges = math.ceil(indices/2)

		interesections = np.zeros((1,2))
		# Find inderecection x_indeces
		if current_edges.shape[1] == 2:
			x_index = np.where(verts_p[1,:] == ymin)
			interesections = [verts_p[0,x_index],verts_p[1,x_index]]
		elif current_edges.shape[1]==4:
			index = 1
			for i in range(3):
				if edges[0,1,i] == edges[1,1,i]:
					interesections = [[edges[0,0,i]],[edges[1,0,i]]]
				else:
					current_edges[index] = i
					index = index + 1
		else:
			interesections[0] = min(x_kmin)
			interesections[1] = max(x_kmax)
		
		insertion_buffer = []
		trash_buffer = []
		tff = TriangleFillingFunction()
		# Scanlines 
		for y in range(ymin,ymax):
			s_points = sorted(np.around(intersections))

			#Start point color 
			index = edges[:,3,current_edges[0]]
			V1 = verts_c[index[0],:]
			V2 = verts_c[index[1],:]
			P1 = verts_p[:,index[0]]
			P2 = verts_p[:,index[1]]
			tff = TriangleFillingFunction()
			A_color = tff.interpolate_vector_color(P1,P2,V1,V2)

			# 1ST POINT LIGHTING 
			N1 = verts_n[:,index[0]]
			N2 = verts_n[:,index[1]]
			normal_A_color = tff.interpolate_vector_color(N1.transpose(),N2.transpose(),V1,V2)

			# Point B coloring 
			index  =  edges[:,3,current_edges[1]]
			V1 = verts_c[index[0],:]
			V2 = verts_c[index[1],:]
			P1 = verts_p[:,index[0]]
			P2 = verts_p[:,index[1]]
			B_color = tff.interpolate_vector_color(P1,P2,V1,V2)

			#Point B lighting

			N1 =  verts_n[:,index[0]]
			N2 =  verts_n[:,index[1]]
			normal_B_color = tff.interpolate_vector_color(N1.transpose(),N2.transpose(),V1,V2)

			# create a and b vectors of points A and B
			a = interesections[0].append(y) 
			b = interesections[1].append(y)

			for x in range(s_points[0],s_points[1]):

				temp_color = tff.interpolate_vector_color(a,b,A_color,B_color)
				temp_normal_color = tff.interpolate_vector_color(a,b,normal_A_color,normal_B_color)

				# Calculate ambient light
				Iamb_temp =  ambient_light(temp_color.transpose(),Ia)
				#Calculate diffuse light
				Id_temp = diffuse_light(bcoords,temp_normal_color.transpose(),temp_color.transpose(),kd,light_positions, light_intensities)
				#Calculate specular light
				Is_temp = specular_light(bcoords,temp_normal_color.transpose(),temp_color.transpose(),ks,n,light_positions, light_intensities)

				X[x,y,:] = temp_color + Iamb_temp.transpose() + Id_temp.transpose()+Is_temp.transpose()


			# DDA algorithm
			for i in range(2):
				if m[current_edges[i]] == float('inf'):
					continue
				else:
					interesections[i] = interesections[i] + 1/m[current_edges[i]]


			# Append edges
			insertion_buffer = np.where(y_kmin == y+1)
			# remove edges
			trash_buffer = np.where(y_kmax == y+1)
			# Update Intersection buffer
			if insertion_buffer.shape[1] == trash_buffer.shape[1]:
				continue
			for i in range(trash_buffer.shape[1]):
				if current_edges[0] ==  trash_buffer[i]:
					current_edges[0] == insertion_buffer[i]
				# Find new intersections
					if edges[0,1,insertion_buffer[i]]==y+1:
						interesections[0] = edges[0,0,insertion_buffer[i]]
					else:
						interesections[0] = edges[1,0,insertion_buffer[i]]
				else:
					current_edges[1] = insertion_buffer[i]
					if edges[0,1,insertion_buffer[i]] == y+1:
						interesections[1] = edges[0,0,insertion_buffer[i]]
					else:
						interesections[1] = edges[1,0,insertion_buffer[i]]
		#loop until all triangles are filled

	Img = X 
	return X
				


 


