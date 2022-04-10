import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from scipy import io.loadmat
from collections import OrderedDict,DefaultDict
from scanline_util import TriangleFillingFunction

def load_binary_data(filename):
	# get the working directory path to __location__ variable
	data = np.load(filename,allow_pickle=True)
	verts2d = data[()]['verts2d']
	vcolors = data[()]['vcolors']
	faces = data[()]['faces']
	depth = data[()]['depth']
	return data,verts2d,vcolors,faces,depth

''' Make this functions to laod mat files and test last year racoon data '''
def load_data_from_mat(filename):
	data = io.loadmat(filename)
	verts2d = data[()]['verts2d']
	vcolors = data[()]['vcolors']
	faces = data[()]['faces']
	depth = data[()]['depth']
    return verts2d, vcolors, faces, depth
   

"""
	Uses the 
"""

def render_image_to_file(filename)




#########################################
#####SCAN LINE ALGORITHM UTILS###########
#########################################

"""
	Compute triangle edge limits based on 
	linear equation of a line 
"""
def edges_limits(verts2d):
	# y2 - y1
	# x2 - x1
	# alpha = (y2-y1)/(x2-x1)
	edge_vertices = np.array([[verts2d[0], verts2d[1]], [verts2d[0], verts2d[2]], [verts2d[1], verts2d[2]]])
'''
	Finds the initial active elements and makes scanline algorithms that are difficult (vertical scan)
	Active edges may be either vertical or have 
	a really high alpha slope curve coefficient
	And some coordinates are on the active edges list.

	Input :
	- active_edges(current)
	- active_nodes(currrent vertices)
	- no_projected(invisible edge that exists in 3d Scene)
'''
def predetermined_active(active_edges, active_nodes, vertices, edge_limits, edge_line_alpha):

	


def determine_active(current_scanline, edge_vertices, edge_limits_y, alpha, active_edges, active_nodes):
		# initialize a set a dynamic array
		new_active_nodes = set()
		# edge_limits_y is a L by 2 enumeration array 
		for i, edge_limit_y in enumerate(edge_limits_y):
			if edge_limit_y[0] == current_scanline
				# isna function check for NaN values and is in pandas and numpy
				# drop NaN values in the predetermined active edges.
				if np.isna(alpha[i]):
					#continue loop for next pair i,edge_limit_y
					continue
				active_edges[i] = True 
				edge_vertices_pos = np.argmin(vertices_of_edge[i,:,1])
				active_nodes[i] = [edge_vertices[i,edge_vertices_pos,0], edge_limits_y[i,0]]
				new_active_nodes.add(i)
			if edge_limit_y[1] == current_scanline:
				active_edges[i] =  False
			if edges_limit_y[0] == current_scanline:
				active_edges[i] == True
		# active_nodes  set() should be updated 
		return active_edges,active_nodes,new_active_nodes

# redetermine active node set based on DDA algorithm where i store the 1/alpha and I do y=y+1 (to find the other active nodes that are missing)
def redetermine_active_nodes_set(active_edges,active_nodes,new_active_nodes,edge_line_alphas):
	for i,alpha in enumerate(edge_line_alphas):
		if active_edges[i]==True and alpha!=0:
			if i not in new_active_nodes:
				active_nodes[i,0] += 1 / edge_line_alphas[i]
				active_nodes[i,1] += 1
	return active_nodes

'''
	Color blending is the function that original implements the scan line
	triangle filling algorithm for predefined scanline and an edge_list
	the limits of the edge the alpha curvature ration of the edge based on
	the active elements of the edge
	Computes the color of the horizontal edges 
	of active vertices 
	edge_list : The indices from the pair of vertices that combined construct an edge
	active_edges: the determined active_edges (updated list)
	active_nodes : the determined active_node (updated list)
	vcolors : the RGB triplet from the vertices of a triangle

'''
def color_blending(current_scanline, edge_list, edge_limits_x, edge_limits_y, alpha, active_edges,active_nodes, vcolors, img):
	active_nodes_color = np.zeros((3,3))
	for i,pointPair in active_nodes:
		if active_edges[i] == True:
			x_edge = np.array(edge_limits_x[i])
			y_edge = np.array(edge_limits_y[i])
			node_PointPair =  edge_list[i]
			C1,C2  = vcolors[node_PointPair[0], vcolors[node_PointPair[1]]]
			coloring_function = TriangleFillingFunction()
			# horizontal edge
			if alpha == 0:
				active_nodes_color[i] = coloring_function.interpolate_color(x_edge[0],x_edge[1],active_nodes[i,0],C1,C2)
				for x in range(x_edge[0], x_edge[1]):
					# import this active_nodes_color to image
					# arround function Evenly round the given number of decimal to get to right pixel (finds nearest integer)
					img[int(np.around(x)),int(np.around(current_scanline))] = coloring_function.interpolate_color()
			# vertical edge
			elif alpha != 0 and np.abs(a) == float('inf'):
				active_nodes_color[i] = coloring_function.interpolate_color(y_edge[0],y_edge[1],current_scanline,C1,C2)
				# color with this only the active nodes (the nodes that are on the triangle)
				img[int(active_nodes[i,0]),int(np.around(current_scanline))]
			# non vertical non horizontal line (the algorithm remains the same for drawing line)
			else:
				start = np.concatenate(x_edge[0],y_edge[0])
				end = np.concatenate(x_edge[1],y_edge[1])
				active_nodes_color[i] = coloring_function.interpolate_color(start,end,scanline,C1,C2)
				img[int(active_nodes[i,0]), int(np.around(current_scanline))] = active_nodes_color[i]
	return img, active_nodes_color
