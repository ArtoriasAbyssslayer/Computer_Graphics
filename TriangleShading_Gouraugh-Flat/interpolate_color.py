import numpy as np
"""
     function that implements the linear interpolation between 2 3D values C1,C2 and C3
     based on 2 points in 2d space of the vertices of a triangle
"""
def interpolateColor(x1,x2,x,C1,C2):
    """
        x1 =  [x1,x2] its a tuple of 2 numbers
        x2 = [x3,x4]
        C1 = [R1,G1,B1]
        C2 = [R2,G2,B2]
        C1 corresponds to the color of x1 point
        C2 corresponds to the color of x2 point
        lest find the color of point x
        gamma parameters are the thalli theorem proportions that are used to compute the linear interpol either from 1->x or 2->x with this order
    """
    if(len(x) == 1):
        if x1[0] == x2[0]:
            C_append = C1.append(C2)
            value = np.mean(C_append)
            return value
        else:   
            gamma= abs(x1[0]-x[0])/abs(x1[0]-x2[0])
            value = gamma*C1 + (1-gamma)*C2
    elif(len(x) == 2):
        if x1[1] == x2[1]:
            C_append = C1.append(C2)
            value = np.mean(C_append)
            return value
        else:
            gamma  = abs(x2[1]-x[1]) / abs(x2[1]-x1[1])
    #There are these complementary coefficients that may be used but abs pretty much make them equal.
        gamma_x2 =  abs(x2[0]-x[0]) / abs(x2[0]-x1[0])
        gamma_y1 =  abs(x2[1]-x[1]) / abs(x2[1]-x1[1])
        gamma_x1 =  abs(x1[0]-x[0]) / abs(x1[0]-x2[0])
        value = abs(gamma_x1-gamma_x2)*C1 + gamma_Y1*C2
    # 1-gamma_x2 is the percentage of the colour similarity of x that is similar to point x2
    # This should be done on y axis as well for 2d point
    # Now the itnerpolation can be done
    
    return value;
