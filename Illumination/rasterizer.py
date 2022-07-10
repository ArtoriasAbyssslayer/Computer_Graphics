import numpy as np
import math

def rasterize(P,M,N,H,W):
	# Place 2D points from camera sensor surface to image frame
	# Pixel Dimensions
	Delta_X = W/N
	Delta_Y = H/M
	num_points = P.shape[0]
	Prast = np.zeros((num_points,2))
	
	# #Move middle axis
	for i in range(num_points):
		P[i,0] = P[i,0]+W/2
		P[i,1] = P[i,1]+H/2
	Prast[:,0] = np.around(P[:,0]/Delta_X)
	Prast[:,1] = np.around(P[:,1]/Delta_Y)
	return Prast