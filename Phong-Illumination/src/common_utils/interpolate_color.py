import numpy as np 
"""
    Color interpolation in 2d space 
    Based on the barycentric coordinates
    
    Inputs:
    p1 : 2D point 1 referenced as x1
    p2 : 2D point 2 referenced as x2
    xy  : 1D or 2D point to be interpolated
    V1 : Color of x1
    V2 : Color of x2
    dim : interpolation space 
    Returns:
    V  : Interpolated color
"""
def interpolateVector(p1,p2,V1,V2,xy,dim):
    p1 = np.array(p1)
    p2 = np.array(p2)
    if dim == 1:
        # c = x - x1 / x2 - x1
        c = (xy - p1)/(p2-p1)
        V = (1-c)*V2 + c*V1
        return V
    elif dim == 2:
        #interpolate using mid point and barycentric coordinates
        pmid = (p1 + p2)/2
        v0 =  p2 - p1
        v1 =  pmid - p1
        v2 = xy - p1
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2) 
        denom = dot00 * dot11 - dot01 * dot01
        # Compute the barycentric coordinates w1,w2,w3 
        if denom == 0:
            w1 = 0            
            w2 = 0            
            w3 = 1
        else: 
            w2 = (dot11 * dot02 - dot01 * dot12) / denom
            w1 = (dot00 * dot12 - dot01 * dot02) / denom
            # w3 represents the weights for V1,V2 and the avg of V1 and V2
            w3 = 1 - w2 - w1
        # interpolate V using barycentric coordinates
        V = w1 * V1 + w2 * V2 + w3 * (V1 + V2) / 2
        "Revised Method"
        c = (p2-xy)/(p2-p1)
        V += np.array(c*V1 + (1-c)*V2)

    else:
        raise ValueError("dim must be 1 or 2")
    return V