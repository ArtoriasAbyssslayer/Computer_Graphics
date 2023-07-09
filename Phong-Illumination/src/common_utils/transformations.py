import numpy as np 
from numpy.linalg import inv


''' Calculate the Rotatation matrix for a given angle and unit vector'''
def getRotmat(theta, u):
    # Convert the unit vector to a numpy array
    u = np.array(u)

    # Ensure that the input vector is a unit vector
    assert np.abs(np.linalg.norm(u) - 1.0) < 1e-6

    # Calculate the components of the rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos = 1 - cos_theta
    u_cross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

    # Calculate the rotation matrix
    R = cos_theta * np.eye(3) + sin_theta * u_cross + one_minus_cos * np.outer(u, u)

    return R

''' Rotate Translate a set of points in 3d space '''
def RotateTranslate(cp,theta,u,A,t):
    R = getRotmat(theta, u)
    # transforms a point cp to cp' as cp' = A*R*cp + t
    cq = [np.copy(cp[j]) for j in range(len(cp))]
    for i in range(len(cp)):
        cq[i] = np.matmul(A,np.matmul(R,cp[i])) + t
    return cq

# def changeCoordinateSystem(cp,R,c0):
#     Rinv = inv(R)
#     dp = [np.copy(cp[j]) for j in range(len(cp))]
#     for i in range(len(cp)):
#         dp[i] = np.matmul(Rinv, cp[i] - c0)
#     # shape prduced numpoints,3,3,1 for some reason
#     return dp

def changeCoordinateSystem(cp,R,c0):
    """
        Remake the function to be more like an homogenous transform
        because problems arise with dimensions when using the old one
    """
    cp = cp.reshape(-1,3)
    cp_augmented_matrix = np.concatenate((cp.T, np.ones((1, cp.shape[0]))), axis=0)
    c0 = c0.reshape(3,1)
    translation = -R.T@c0
    homogenus_transform = np.concatenate([R.T,translation],1)
    dp = np.matmul(homogenus_transform,cp_augmented_matrix).T
    return dp[:, 0:3] # keep only the first 3 rows, the last row is the homogenous coordinate which is always 1



