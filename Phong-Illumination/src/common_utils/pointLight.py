import numpy as np 

class PointLight:
    
    def __init__(self,PhongMaterial,l_pos,l_int):
        self.PhongMaterial = PhongMaterial
        self.l_pos = l_pos
        self.l_int = l_int
        assert self.l_int.all() <= 1 and self.l_int.all() >=0 , "Light intesity values must be in range [0,1]"
    





