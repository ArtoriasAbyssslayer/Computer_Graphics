import numpy as np

from src.common_utils.phongMaterial import PhongMaterial
from src.common_utils.pointLight import PointLight

class PhongIlluminationModel(PhongMaterial,PointLight):

    def __init__(self,ka,kd,ks,n_phong,l_pos,l_int):
        self.PhongMaterial = PhongMaterial(ka,kd,ks,n_phong)
        self.PointLight = PointLight(self.PhongMaterial,l_pos,l_int)
        
    def ambient_light(self,ka,I_a):
        return ka*I_a

    ''' Implement Lambertian reflection '''

    def diffuse_lights(self, P, N, color, kd, l_p, l_i):
        # Parameters:
        #    - P vector containing the 3-D coordinates of the point P
        #    - N normal-vector of the surface that p lies
        #    - color = [c_r,c_g,c_b]' colour components of P point range [0-1]
        #    - l_p = light_positions list of [3x1] poisitions of lights
        #    - l_i = light_intensities of each light source [I_r,I_g,I_b]x3

        # get the unitary vector
        N_u = N/np.absolute(N)
        

        # get unitary L vector
        # L vector is the vector of the light beam that reflects on the surface of P
        L = l_p - P
        L_u = L/np.linalg.norm(L)

        # Compute the dot product of the unitary vector to find the cosine
        # of the incident rays
        cosb = np.dot(L_u,N_u)
        I = kd*cosb*l_i
        return color*I

    # specular light I = k_d*I_0(R*V)^N
    # I = specular_light(P, N, color, cam_pos, ks , n, light_positions, light_intensities)
    def specular_light(self, P, N, color, cam_pos, k_s, n, l_p, l_i):
        N_u = N/np.absolute(N)
        # calculate R vector
        L = P - l_p
        L_u = L/np.absolute(L)
        R = 2*N_u*np.dot(L_u,N_u)-L_u
        V = P - cam_pos
        V_u = V/np.absolute(V)
        # R*V
        sl_coeff = np.dot(R, V_u)
        specular_light_val = sl_coeff**n
        return specular_light_val*k_s*l_i*color
    
    """
        Accumulated Phong Lighting method 
    """
    def light(self,point,normal,vcolor,cam_pos,light_amb):
        # Initialize secondary variables from objects passed as arguments 
        lights = self.PointLight
        mat = self.PhongMaterial
        # Initialize the ambient light
        ambient_lighting = self.ambient_light(mat.ka,light_amb)
        # Initialize the diffuse light
        diffuse_lighting = self.diffuse_lights(point,normal,vcolor,self.PhongMaterial.kd,lights.l_pos,lights.l_int)
        # Initialize the specular light
        specular_lighting = self.specular_light(point,normal,vcolor,cam_pos,mat.ks,mat.n_phong,lights.l_pos,lights.l_int)
        
        
        # Compute the total light
        light = ambient_lighting + diffuse_lighting + specular_lighting
        
        
        
        return light