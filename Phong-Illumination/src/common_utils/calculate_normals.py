import numpy as np 

def calculate_normals(vertices, face_indices):
    #allocate space for normal vectors
	normals = np.zeros(len(vertices))

	points_1 = face_indices[0,:]
	points_2 = face_indices[1,:]
	points_3 = face_indices[2,:]

	#Select each triangle vertices
	A = vertices[:,points_1-1]
	B = vertices[:,points_2-1]
	C = vertices[:,points_3-2]

	#3 by N matrices 
	AB = B-A 
	BC = C-B

	# Find Normal vector with cross product
	crossABBC  = np.cross(AB,BC)
	norm_ABBC =  np.linalg.norm(crossABBC,ord=1,axis=None,keepdims=False)

	# The normal vector is
	N = crossABBC/norm_ABBC

	#Find normals for every vertex the triangles belong
	
	for i in face_indices: 
		triangles = vertices[i]
		# calculate the normal of each face
		face_N = np.cross(triangles[1]-triangles[0],triangles[2] - triangles[1])
		face_N = face_N/np.linalg.norm(face_N)
		# Add the normal vector of the triangles
		normals[i] += face_N


	Normals_norm =  np.linalg.norm(normals)
	normals = normals/Normals_norm



	return  normals
    