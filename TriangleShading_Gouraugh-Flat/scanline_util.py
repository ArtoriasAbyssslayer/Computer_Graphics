import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
import os
import auxiliary_funcs as aux 

class TriangleFillingFunction:
	def __init__(self, return_value):
		self.return_value = return_value



    def interpolate_color(self,x1,x2,C1,C2):
		"""

			Function that implements the linear interpolation between 2 3D values C1,C2 and C3
			based on 2 points in 2d space of the vertices of a triangle

	        x1 =  [x1,x2] its a tuple of 2 numbers
	        x2 = [x3,x4]
	        C1 = [R1,G1,B1]
	        C2 = [R2,G2,B2]
	        C1 corresponds to the color of x1 point
	        C2 corresponds to the color of x2 point

	        gamma parameters are the thalli theorem proportions that are used to compute the linear interpol either from 1->x or 2->x with this order
	    """
		if(x1[0]==x2[0]):
			#interpolate according y_coordinate
			gamma = abs(x2[1]-x[1])/abs(x2[1]-x1[1])
			color_value = np.array(gamma*C1 + (1-gamma)*C2)
		elif(x1[1]==x2[1]):
			#interpolate according to x_coordinate
			gamma = abs(x2[0]-x[0])/abs(x2[0]-x1[0])
			color_value = np.array(gamma*C1+(1-gamma)*C2)
		else:

			k1 = math.sqrt(pow(x1[0],2)+pow(x1[1],2))
			k2 = math.sqrt(pow(x2[0],2)+pow(x2[1],2))
			k3 = math.sqrt(pow(x[0],2)+pow(x[1],2))
			lambda1 = abs(k2-k3)/abs(k1-k2)
			lambda2 = abs(k1-k3)/abs(k1-k2)
			d = lambda1*C2 + (1-lambda1)*C1
			p = lambda2*C1 + (1-lambda2)*C2
			# at this point I assume p==d but I will take their mean
			interpolation_arr = np.array(d,p)
			color_value = np.mean(interpolation_arr, axis=0)

    # 1-gamma_x2 is the percentage of the colour similarity of x that is similar to point x2
    # This should be done on y axis as well for 2d point
    # Now the interpolation can be done
	 # Triangle shading algorithm
		""" 
			Arguments
			# 1. img : the image 2Dx3 with triangles
			# 2. verts2d : interger matrix 3x2 [in each row contained the 2D coordinates of each vertex of the triagnle]
			# 3. vcolors : matrix
			# 4. 
			return_value:
			- s_triangle : shaded triangle 

		"""
	def shade_triangle(self, img, verts2d, vcolors, shade_t):
			# temporary variables
			"""
			 # 1. active_edges -> A list of 
			 # 2. active_nodes ->

			"""
			if(shade_t = "gouraud"):
				estimated_color = np.array(np.mean())

			elif(shade_t = "flat"):

			else:
				print("Not valid shading algorithm.")
			return Y


	"""
		render function colors every triangle in a loop using predetermined
		Parameters:
		- verts2D:  Lx2 matrix with the coordinates of every vertex of the graph
		- faces: Kx3 matrix with the vertices of K triangles 
		- depth: Lx1 matrix with the depth of every vertex in initial 3D space (before image is projected)
		return value:
		- img:	MxNx3 image with colors
	"""
	def render(verts2d,faces,vcolors,depth,shade_t):
		if(shade_t != "flat") || (shade_t != "gouraud"):
			print("Not valid shading algorithm")
		if(m < 0 || n < 0):
			print("Not valid graphic image dimensions")
		'''Define a white image that will be rendered(filled with color) '''
		img = np.ones((m,n,3));
		for faces in faces:
			for vertex in face:
				triangle_depths[vertex] = depth[face]
		sorted_depths = np.sort(depths,axis=0)
		reversed_depths = sorted_depths[::-1]
		# find vcolors for descending depths sorted triangles 
		for st in reversed_depths:
			v_colors_sorted_triang =  faces[st]
			if(shade_t ==  'flat')
				img = shade_triangle(img,reversed_depths,v_colors_sorted_triang,'flat');
			if(shade_t == 'gouraud')
				img = shade_triangle(img,reversed_depths,v_colors_sorted_triang,'gouraud');
		return img
