import numpy as np 

''' Implement Lambertian reflection ''' 
def diffuse_light(P,N,color,k_d,l_p,l_i):
#	Parameters:
#    - P vector containing the 3-D coordinates of the point P
#    - N normal-vector of the surface that p lies 
#    - color = [c_r,c_g,c_b]' colour components of P point range [0-1]
#    - k_d Phong coefficient of calculating diffuse light reflection with this model
#    - l_p = light_positions list of [3x1] poisitions of lights
#    - l_i = light_intensities of each light source [I_r,I_g,I_b]x3

	# get the unitary vector 
	N_u = N/np.absolute(N)

	# get unitary L vector 
	# L vector is the vector of the light beam that reflects on the surface of P
	L = P - l_p
	L_u = L/np.absolute(L)

	#Compute the dot product of the unitary vector to find the cosine
	# of the incident rays
	cosb = np.dot(N_u,L_u)

	I = k_d*cosb*l_i
	return I 
	