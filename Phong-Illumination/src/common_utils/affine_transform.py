import numpy as np 
def affine_transform(cp,RotationMatrix,translateVec):
    """
        Applies affine transform as 
        T = [R t]
    
        
    """
    ones_vec = np.ones((cp.shape[1],1))
    cp_augmented_matrix = np.concatenate((cp.T,ones_vec), axis=1)
    translateVec = np.reshape(translateVec,(3,1))
    homogenous_transform = np.concatenate((RotationMatrix,  translateVec),axis=1)
    homogenous_transform = np.concatenate((homogenous_transform, np.zeros((1, 4))), axis=0)
    homogenous_transform[3,3] = 1
    transformed_points = np.matmul(homogenous_transform,cp_augmented_matrix.T)
    transformed_points = transformed_points.T[:, 0:3].reshape(-1,3)
    return transformed_points# keep only the first 3 rows, the last row is the homogenous coordinate which is always 1
