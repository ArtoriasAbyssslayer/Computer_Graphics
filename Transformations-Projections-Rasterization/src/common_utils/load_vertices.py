import numpy as np 
import imageio as io


def load_numpy_data(filename):
    """
    
        Loads numpy array for Project 2
        
        Containing Vals:
        verts3d,vcolors,faces,u,ck,cu,cv,t1,t2,phi
        
        p3d: 3D coordinates of all vertices of the triangles
        vcolors: The RGB values for each vertex
        faces: Indices of each triangle 3-verts tuple
        u: axis along which the rotation is made
        ck: the pointing coordinates of the camera
        # view coordinate system
        cu: the up-vector of the camera 
        cv: the camera vector perpendicular to the up vector 
        # Transformation parameters
        t1: translation displacment vector 1
        t2: translation displacement vector 2
        phi: rotation angle 
    """
    
    data = np.load(filename,allow_pickle=True).tolist()
    
    data = dict(data)
    p3d = np.array(data['verts3d'])
    faces = np.array(data['faces'])
    vcolors = np.array(data['vcolors'])
    u = data['u']
    ck = data['c_lookat']
    cu = data['c_up']
    cv = data['c_org']
    t1, t2 = data['t_1'], data['t_2']
    phi = data['phi']
    # convert to numpy array with float32 type for better performance 
    # p3d, faces, vcolors = np.array(np.array(p3d), dtype=np.float32), np.array(np.array(faces), dtype=np.int32), np.array(np.array(vcolors), dtype=np.float32)
  
    return p3d, faces, vcolors, u, ck, cu, cv,t1, t2, phi

    
def load_binary_data(filename):

    data = np.load(filename, allow_pickle=True).tolist()
    data = dict(data)
    verts2d = np.array(data['verts2d'])
    vcolors = np.array(data['vcolors'])
    faces = np.array(data['faces'])
    depth = np.array(data['depth'])
    # Turn the image by 90 degrees
    verts2d_final = np.zeros((verts2d.shape[0], 2))
    verts2d_final[:, 0] = verts2d[:, 1]
    verts2d_final[:, 1] = verts2d[:, 0]

    return verts2d, vcolors, faces, depth


def load_data_from_mat(filename):
    data = io.loadmat(filename)
    verts2d = np.array(data['vertices_2d'] - 1)
    vcolors = np.array(data['vertex_colors'])
    faces = np.array(data['faces'] - 1)
    depth = np.array(data['depth']).T[0]
    return verts2d, vcolors, faces, depth

