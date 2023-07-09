import numpy as np 

def calculate_normals(vertices, face_indices):
    #allocate space for normal vectors
    vertices = vertices.T
    
    normals = np.zeros((vertices.shape[0], 3))
    
    for indices in face_indices:
        triangle_vertices = vertices[indices,:]
        face_normal = np.cross(triangle_vertices[1] - triangle_vertices[0], triangle_vertices[2] - triangle_vertices[1])
        normals[indices] += face_normal

    for i, normal in enumerate(normals):
        normals[i] = normal / np.linalg.norm(normal)
    return normals
    
	
	