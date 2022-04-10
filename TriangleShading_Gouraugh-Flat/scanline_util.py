import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
import os
import auxiliary_funcs as aux


class TriangleFillingFunction:
	def __init__(self):
		pass
	def interpolate_color(self,x1,x2,C1,C2):
		# '''
		# 	Function that implements the linear interpolation between 2 3D values C1,C2 and C3
		# 	based on 2 points in 2d space of the vertices of a triangle
		#
		#     x1 =  [x1,x2] its a tuple of 2 numbers
		#     x2 = [x3,x4]
		#     C1 = [R1,G1,B1]
		#     C2 = [R2,G2,B2]
		#     C1 corresponds to the color of x1 point
		#     C2 corresponds to the color of x2 point
		#
		#     gamma parameters are the thalli theorem proportions that are used to compute the linear interpol either from 1->x or 2->x with this order
		# '''
		if x1[0]==x2[0]:
			# interpolate according y_coordinate
			gamma = abs(x2[1]-x[1])/abs(x2[1]-x1[1])
			color_value = np.array(gamma*C1 + (1-gamma)*C2)
		elif x1[1]==x2[1]:
			# interpolate according to x_coordinate
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
			# 4. shade_t : shading algorithm
			This function uses essential functions created in auxialiary_funcs.py and implement core scanline utils


		"""
	def shade_triangle(self, img, verts2d, vcolors, shade_t):
			# temporary variables
			"""
			 # 1. active_edges -> A list of binary values that determine that the scanline passes from this point in
			 # 2. active_nodes ->

			"""
			if shade_t == "flat":
				estimated_color = np.array(np.mean(vcolors,axis=0))
				if(verts2d == verts2d[0]).all:
					img[int(verts2d[0, 0]), int(verts2d[0, 1])] = estimated_color
					return img
				# find edges attributes
				edges_vertices,x_bounds,y_bounds,alpha = aux.find_edge_bounds(verts2d)

				x_min = int(np.ndarray.min(x_bounds))
				x_max = int(np.ndarray.max(x_bounds))
				y_min = int(np.ndarray.min(y_bounds))
				y_max = int(np.ndarray.max(y_bounds))

				# find active edges that are predetermined

				active_edges = np.array([False,False,False])
				active_nodes = np.zeros((3,2))

				edge_tuple =  {0:[0,1], 1:[0,2], 2:[1,2]}
				# active_edges  assumed as current_scanline
				active_edges,active_nodes,hidden_edge = aux.predetermined_active(active_edges,active_nodes,edge_tuple,edge_limi)
				if hidden_edge == True:
					return img
				# scanline algorithm
				for y in range(y_min,y_max):
					# update the active edges based on scanline algorithm
					active_edges,active_nodes,new_active_nodes = aux. determine_active(y, edge_vertices,y_bounds, alpha, active_edges, active_nodes)
					# redetermine the nodes based on DDA
					active_nodes = aux.redetermine_active_nodes_set(active_edges,active_nodes,new_active_nodes,edge_line_alphas)
					# use colour blending
					img, active_nodes_color = color_blending(y, edge_vertices, x_bounds, y_bounds, alpha, active_edges,active_nodes, vcolors, img)
					intersection_counter = 0;
					for x in range(x_min, x_max):
						if x == np.around(active_nodes[active_edges][:, 0])!=0 :
							intersection_counter += 1
						if intersection_counter%2 != 0:
							img[x,y] = estimated_color
						elif y == ymax :
							if (np.arround(active_nodes[active_edges][:,0]) > 0):
								img[x, y] = new_color

					active_edges,active_nodes,new_active_nodes = aux. determine_active(y, edge_vertices,y_bounds, alpha, active_edges, active_nodes)
					active_nodes = aux.redetermine_active_nodes_set(active_edges,active_nodes,new_active_nodes,edge_line_alphas)

					return img
			elif shade_t == "gouraud":
				estimated_color = np.array(np.mean(vcolors, axis=0))
				if (verts2d == verts2d[0]).all():
					img[int(verts2d[0, 0]), int(verts2d[0, 1])] = estimated_color
					return img

				# find edges attributes
				edges_vertices,x_bounds,y_bounds,alpha = aux.find_edge_bounds(verts2d)

				x_min = int(np.ndarray.min(x_bounds))
				x_max = int(np.ndarray.max(x_bounds))
				y_min = int(np.ndarray.min(y_bounds))
				y_max = int(np.ndarray.max(y_bounds))

				# find active edges that are predetermined

				active_edges = np.array([False,False,False])
				active_nodes = np.zeros((3,2))

				edge_tuple =  {0:[0,1], 1:[0,2], 2:[1,2]}
				# active_edges  assumed as current_scanline
				active_edges,active_nodes,hidden_edge = aux.predetermined_active(active_edges,active_nodes,edge_tuple,edge_limi)
				if hidden_edge == True:
					return img
				# scanline algorithm
				for y in range(y_min,y_max):
					# update the active edges based on scanline algorithm
					active_edges,active_nodes,new_active_nodes = aux. determine_active(y, edge_vertices,y_bounds, alpha, active_edges, active_nodes)
					# redetermine the nodes based on DDA
					active_nodes = aux.redetermine_active_nodes_set(active_edges,active_nodes,new_active_nodes,edge_line_alphas)
					# use colour blending
					img, active_nodes_color = color_blending(y, edge_vertices, x_bounds, y_bounds, alpha, active_edges,active_nodes, vcolors, img)

					# now interpolate between the scanline pixels to reconstruct the image
					x_left, idx_left = np.min(active_nodes[active_edges, 0]), np.argmin(active_nodes[active_edges, 0])
					x_right, idx_right = np.max(active_nodes[active_edges, 0]), np.argmax(active_nodes[active_edges, 0])
					C1, C2 = active_nodes_color[active_edges][idx_left], active_nodes_color[active_edges][idx_right]

					intersection_counter = 0
					for x in range(x_min,xmax):
						if x == np.around(active_nodes[active_edges][:,0]):
							intersection_counter += np.around(active_nodes[active_edges][:,0]);
						elif intersection_counter % 2 and int(np.around(x_left)) != int(np.around(x_right)):
							img[x,y] = estimated_color

						# update edges
						active_edges,active_nodes,new_active_nodes = aux. determine_active(y, edge_vertices,y_bounds, alpha, active_edges, active_nodes)

						# update nodes
						active_nodes = aux.redetermine_active_nodes_set(active_edges,active_nodes,new_active_nodes,edge_line_alphas)

					return img
			else:
				print("Not valid shading algorithm.")



	"""
		render function colors every triangle in a loop using predetermined
		Parameters:
		- verts2D:  Lx2 matrix with the coordinates of every vertex of the graph
		- faces: Kx3 matrix with the vertices of K triangles
		- depth: Lx1 matrix with the depth of every vertex in initial 3D space (before image is projected)
		return value:
		- img:	MxNx3 image with colors
	"""
	def render(self,verts2d,faces,vcolors,depth,M,N,shade_t):
		if shade_t != "flat" or shade_t != "gouraud":
			print("Not valid shading algorithm")
		if M < 0 or N < 0:
			print("Not valid graphic image dimensions")
		'''Define a white image that will be rendered(filled with color) '''
		img = np.ones((M,N,3));
		triangle_depths = np.array(np.mean(depth[faces], axis = 1))
		reversed_depths =  list(np.flip(np.argsort(triangle_depths)))
		# find vcolors for descending depths sorted triangles
		for st in reversed_depths:
			v_colors_sorted_triang = faces[st]
			if shade_t == 'flat':
				img = self.shade_triangle(img,reversed_depths,v_colors_sorted_triang,'flat');
			if shade_t == 'gouraud':
				img = self.shade_triangle(img,reversed_depths,v_colors_sorted_triang,'gouraud');
		return img
