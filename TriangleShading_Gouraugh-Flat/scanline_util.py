import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


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
			lambda1 = abs(x2[1]-x[1])/abs(x2[1]-x1[1])
			lambda2 = abs(x2[0]-x[0])/abs(x2[0]-x1[0])
			interpolation_arr = np.array([lambda1*C1 + (1-lambda1)*C2],[lambda2*C1 + (1-lambda2)*C2])
			color_value = np.mean(interpolation_arr, axis=0)
			
    # 1-gamma_x2 is the percentage of the colour similarity of x that is similar to point x2
    # This should be done on y axis as well for 2d point
    # Now the itnerpolation can be done



	def shade_triangle(self, img, verts2d, vcolors, shade_t):
			# Triangle shading algorithm
			''' Arguments
			# 1. img : the image 2Dx3 with triangles
			# 2. verts2d : interger matrix 3x2 [in each row contained the 2D coordinates of each vertex of the triagnle]
			# 3. vcolors : matrix
			'''
			# get the working directory path to __location__ variable
		    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
		    d = os.path.join(__location__,'./GHW_assets/hw1.npy')
			data = np.load(d,allow_pickle=True)
			verts2d = data[()]['verts2d']
			vcolors = data[()]['vcolors']
			faces = data[()]['faces']
			depth = data[()]['depth']
			if(shade_t = "gouraud"):

			elif(shade_t = "flat"):

			else:
				print("Not valid shading algorithm.")
			return Y

	def render(verts2d,faces,vcolors,depth,shade_t):
		if(shade_t != "flat") || (shade_t != "gouraud"):
			print("Not valid shading algorithm")
		if(m < 0 || n < 0):
			print("Not valid graphic image dimensions")
		'''Define image that will be rendered '''
		img = np.ones((m,n,3))
