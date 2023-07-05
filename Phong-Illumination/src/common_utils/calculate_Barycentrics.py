import numpy as np
def calculateBarycentricCoordinates(vertices,face_indices):
    barycentrics = np.zeros(len(vertices)/3)
    for i in face_indices:
        triangles = vertices[i]
        # calculate the barycentric of each triangle of eac face
        face_barycentric = np.cross(triangles[1]-triangles[0],triangles[2] - triangles[1])
        face_barycentric = face_barycentric/np.linalg.norm(face_barycentric)
        barycentrics[3*i] += face_barycentric
    barycentrics_norm =  np.linalg.norm(barycentrics)
    barycentrics = barycentrics/barycentrics_norm
    return barycentrics
    