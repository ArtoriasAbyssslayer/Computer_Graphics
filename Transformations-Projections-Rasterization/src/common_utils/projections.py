import numpy as np 
from src.common_utils.transformations import changeCoordinateSystem


''' Project a set of points onto a plane - perpspective projection '''
def pinHole(f,cv,cx,cy,cz,p3d):
    
    R_wcs = np.array([cx, cy, cz]) 
    R = np.transpose(R_wcs) # projects base vectors of CCS onto the base vectors of WCS 
    N = p3d.shape[0] # number of vertices to be projected
    verts2d = np.zeros((N,2))
    depths = np.zeros((N,1))
    pCCS = changeCoordinateSystem(p3d, R, cv) # change coordinate system from WCS to CCS
    for i in range(N):
        verts2d[i][0] = f*pCCS[i][0]/pCCS[i][2] # x_proj = f*x/Z,
        verts2d[i][1] = f*pCCS[i][1]/pCCS[i][2] #  y_proj = f*y/Z]
    depths = pCCS[:,2] # depth = Z
    return verts2d, depths

# Makes the projection along camera pointing vector clookat - corg 
def cameraLookingAt(f, corg, clookat, cup, verts3d):
    zCCSBaseVector = clookat - corg # zCCSBaseVector is the camera pointing vector
    zCCSBaseVector = np.array(zCCSBaseVector)/np.linalg.norm(zCCSBaseVector) # z unit vector
    t = cup - np.dot(cup.T, zCCSBaseVector) * zCCSBaseVector # calculate translation vector
    yCCSBaseVector = t/np.linalg.norm(t) # y unit 
    xCCSBaseVector = np.cross(yCCSBaseVector.T, zCCSBaseVector.T) # x unit cross product  
    xCCSBaseVector = xCCSBaseVector/np.linalg.norm(xCCSBaseVector) 
    cv = np.copy(corg)
    cx, cy, cz = xCCSBaseVector.T, yCCSBaseVector, zCCSBaseVector
    p3d = verts3d 
    verts2d , depths = pinHole(f, cv, cx, cy, cz, p3d)
    return verts2d, depths

    