import numpy as np
def calculateBarycentricCoordinates(vertices,face_indices):
    barycentrics = np.zeros((vertices.shape[1],3))
    for i in face_indices:
        triangles = vertices[:,i]
        # calculate the barycentric of each triangle of eac face
        face_barycentric = (triangles[:,i] + triangles[:,i+1] + triangles[:,i+2])/3
        face_barycentric = face_barycentric.transpose()
        barycentrics[i,:] = face_barycentric
    barycentrics_norm =  np.linalg.norm(barycentrics)
    barycentrics = barycentrics/barycentrics_norm
    
    return barycentrics
    