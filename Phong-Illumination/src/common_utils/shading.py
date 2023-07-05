import numpy as np
from src.common_utils.interpolate_color import interpolateVector 
from src.common_utils.scanlineUtilities import *
from src.common_utils.lighting import PhongIlluminationModel
"""
    shadeFlat shades a single triangle 
    with flat shading 
    Vertices : the vertices of the tTraceback (most recent call last):
    Vcolors : the colors of each vertex 3x3 array
    canvas : the original canvas to draw on 
"""



def shadeGouraud(verts2d, vcolors, img):
    """Renders the image, using interpolate colors to achieve smooth color transitioning

    Parameters
    ----------
    verts2d : Lx2 matrix containing the coordinates of every vertex (L vertices)
    vcolors : Lx3 matrix containing the RGB color values of every vertex
    img : MxNx3 image matrix

    Returns
    -------
    img : updated MxNx3 image matrix
    """
    if (verts2d == verts2d[0]).any():
        img[int(verts2d[0, 0]), int(verts2d[0, 1])] = np.mean(vcolors, axis=0)
        return img
    
    
    vertices_of_edge, x_limits_of_edge, y_limits_of_edge, sigmas_of_edges = compute_edge_limits(verts2d)

    # find min/max x and y
    x_min, x_max = int(np.amin(x_limits_of_edge)), int(np.amax(x_limits_of_edge))
    y_min, y_max = int(np.amin(y_limits_of_edge)), int(np.amax(y_limits_of_edge))

    # find initial active edges for y = 0
    active_edges = np.array([False, False, False])
    active_points = np.zeros((3, 2))

    node_combination_on_edge = {0: [0, 1],
                                1: [0, 2],
                                2: [1, 2]
                                }

    active_edges, active_points, is_invisible = initial_active_elements(active_edges, active_points, vertices_of_edge,
                                                                           y_limits_of_edge, sigmas_of_edges)
    if is_invisible:
        return img

    for y in range(y_min, y_max):
        active_edges, active_points, updated_nodes = update_active_edges(y, vertices_of_edge, y_limits_of_edge,
                                                                            sigmas_of_edges, active_edges, active_points)
        active_points = update_active_nodes(sigmas_of_edges, active_edges, active_points, updated_nodes)

        img, active_nodes_color = color_interp(y, node_combination_on_edge, x_limits_of_edge, y_limits_of_edge,
                                                    sigmas_of_edges, active_edges, active_points, vcolors, img)

        x_left, idx_left = np.min(active_points[active_edges, 0]), np.argmin(active_points[active_edges, 0])
        x_right, idx_right = np.max(active_points[active_edges, 0]), np.argmax(active_points[active_edges, 0])
        c1, c2 = active_nodes_color[active_edges][idx_left], active_nodes_color[active_edges][idx_right]

        cross_counter = 0
        for x in range(x_min, x_max + 1):
            cross_counter += np.count_nonzero(x == np.around(active_points[active_edges, 0]))
            if cross_counter % 2 != 0 and int(np.around(x_left)) != int(np.around(x_right)):
                img[x, y] = InterpolateVector(int(np.around(x_left)), int(np.around(x_right)), x, c1, c2,dim=2)

    return img


def shadeFlat(verts2d, vcolors, img):
    """Renders the image, using a single color for each triangle

    Parameters
    ----------
    verts2d : Lx2 matrix containing the coordinates of every vertex (L vertices)
    vcolors : Lx3 matrix containing the RGB color values of every vertex
    img : MxNx3 image matrix

    Returns
    -------
    img : updated MxNx3 image matrix
    """
    new_color = np.array(np.mean(vcolors, axis=0))
    if (verts2d == verts2d[0]).all():
        img[int(verts2d[0, 0]), int(verts2d[0, 1])] = new_color
        return img

    # compute edge limits and sigma
    vertices_of_edge, x_limits_of_edge, y_limits_of_edge, sigmas_of_edges = compute_edge_limits(verts2d)

    # find min/max x and y
    x_min, x_max = int(np.amin(x_limits_of_edge)), int(np.amax(x_limits_of_edge))
    y_min, y_max = int(np.amin(y_limits_of_edge)), int(np.amax(y_limits_of_edge))

    # find initial active edges for y = 0
    active_edges = np.array([False, False, False])
    active_points = np.zeros((3, 2))

    active_edges, active_points, is_invisible = initial_active_elements(active_edges, active_points, vertices_of_edge,
                                                                           y_limits_of_edge, sigmas_of_edges)
    if is_invisible:
        return img

    # dsp.show_vscan(y_min, active_edges, active_points, vertices_of_edge)
    for y in range(y_min, y_max + 1):
        # dsp.show_vscan(y, active_edges, active_points, vertices_of_edge)
        cross_counter = 0
        for x in range(x_min, x_max + 1):
            cross_counter += np.count_nonzero(x == np.around(active_points[active_edges][:, 0]))
            if cross_counter % 2 != 0:
                img[x, y] = new_color
            elif y == y_max and np.count_nonzero(x == np.around(active_points[active_edges][:, 0])) > 0:
                img[x, y] = new_color

        active_edges, active_points, updated_nodes = update_active_edges(y, vertices_of_edge, y_limits_of_edge,
                                                                            sigmas_of_edges, active_edges, active_points)
        active_points = update_active_nodes(sigmas_of_edges, active_edges, active_points, updated_nodes)
    return img


"""
    Remake The shading functions based on Phong Shading system
    
"""
def shadePhong(verts_p,
               verts_n,
               verts_c,
               bcoords,
               cam_pos,
               mat,
               lights,
               light_amb,
               img):
    """
        verts_p: Kx2 matrix which contains all the 2D coordinates of the vertices of each vertex(K vertices)
        verts_n: The normal vectors of every traingle
        verts_c: Vertex color Kx3 matrix (K vertices)
        b_coords: The baricentric coordinates of the triangle for every triangle
        cam_pos: Camera position 3x1 coords (WCS)
        mat: Phong material object containing the ka,kd,ks coeff for Ambient,Diffuse and Spectral Lighting
        lights: PointLight list of objects objects 
        lights_amb: [I_r,I_g,I_b]T 3x1 vector of the light intensities
        img: The canvas of the image that is updated
    """

    
    
    vertices_of_edge, x_limits_of_edge, y_limits_of_edge, sigmas_of_edges = compute_edge_limits(verts_p)
    x_min, x_max = int(np.amin(x_limits_of_edge)), int(np.amax(x_limits_of_edge))
    y_min, y_max = int(np.amin(y_limits_of_edge)), int(np.amax(y_limits_of_edge))
    
    # Active Elements of the triangle
    # Active Points: Points where we estimate the colro
    # Active Edges: Edges of the trinagles that are used for interpolating the color 
    
    active_edges = np.array([False,False,False])
    active_points = np.zeros((3,2))
    
    # nodes-edges dict 
    nodes_on_edge = {0: [0,1],
                     1: [0,2],
                     3: [1,2]}
    
    # Get a buffer for the normals on active nodes
    active_normals_on_edge = np.zeros((3,3))
    
    # Create Phong Illumination Objects for every Light Source :=: PointLightObject
    phongLightingSource = [None]*len(lights)
    for i,_ in enumerate(lights):
        phongLightingSource[i] = PhongIlluminationModel(mat.ka,mat.kd,mat.ks,mat.n_phong,lights[i].l_pos,lights[i].l_int)
    
    ''' Scanline for estimating normals'''
    for y in range (y_min,y_max):
        # every loop := every scanline update the active elements
        active_edges, active_points, updated_nodes = update_active_edges(y,vertices_of_edge,y_limits_of_edge,active_edges,active_points)
        updated_nodes =update_active_nodes(sigmas_of_edges,active_points,updated_nodes)
        # update the canvas := img variable
        img, active_points_color = color_interp(y,nodes_on_edge,x_limits_of_edge,y_limits_of_edge,sigmas_of_edges,active_edges,active_points,verts_c,img)
        
        # Apply scanline second step of interpolation for the active nodes
        for i,v in enumerate(active_points):
            # Calculate the x bounds
            x_edges = np.array(x_limits_of_edge[i]) 
            y_edges = np.array(y_limits_of_edge[i])
            # select edge-nodes combination 
            pairs_of_nodes = nodes_on_edge[i]
            # n1,n2 = normal vectors on edge - To be used in interpolateVector in order 
            # to seek the normal on acvie node
            n1,n2  = verts_n(pairs_of_nodes[0],pairs_of_nodes[1])
            # select the normals for the corespnding pairs_of_nodes 
            
            # Case1 - Horizontal Edge := use x_bounds to interpolate no problem
            if sigmas_of_edges[i] == 0:
                active_normals_on_edge = InterpolateVector(int(np.around(x_edges[0])),int(np.around(x_edges[1])),n1,n2,active_points[i,0],dim=2)
                # normalize estimated normals
                active_normals_on_edge[i] = active_normals_on_edge[i] / np.linalg.norm(active_normals_on_edge[i])
            # Case2 - Vertical Edge  := use y bound to interpolate
            elif np.abs(sigmas_of_edges[i] == float('inf')):
                active_normals_on_edge[i] = InterpolateVector(int(np.around(y_edges[0])),int(np.around(x_edges[1])),n1,n2,y,dim=2)
                active_normals_on_edge[i] = active_normals_on_edge[i] / np.linalg.norm(active_normals_on_edge[i])
            # Case3 - Edge alpha in R* := we use the scanline conventio so ys again
            else:
                active_normals_on_edge[i] = InterpolateVector(int(np.around(y_edges[0])),int(np.around(y_edges[1])),n1,n2,y,dim=2)
                active_normals_on_edge[i] = active_normals_on_edge[i] / np.linalg.norm(active_normals_on_edge[i])
                
                
                
                
            '''Scanline for Color Vectors'''
            x_left_bound , index_l = np.min(active_points[active_edges, 0]), np.argmin(active_points[active_edges, 0])
            x_right_bound, index_r = np.max(active_points[active_edges, 0]), np.argmax(active_points[active_edges, 0])
            c1 = active_points_color[active_edges][index_l]
            c2 = active_points_color[active_edges][index_r]
            cross_counter = 0  
            # knowing th normals on edges we can calculate the lighting
            # and add it to the interpolated color 
            for x in range(x_min,x_max+1):
                cross_counter += np.count_nonzero(x == np.around(active_points[active_edges,0]))
                if cross_counter % 2 != 0:
                    # out of bounds
                    if x < img.shape[0] and x >=0 or y < img.shape[1] and y >=0:
                        normal = InterpolateVector(int(np.around(x_left_bound)),int(np.around(x_right_bound)),n1,n2,x,dim=2)
                        normal = normal / np.linalg.norm(normal)
                        updated_color = InterpolateVector(int(np.around(x_left_bound)), int(np.around(x_right_bound)),c1,c2,x,dim=2)
                        for i,_ in enumerate(lights):
                            img[x,y] += phongLightingSource[i].light(bcoords,normal,updated_color,cam_pos,light_amb)
                            
            
    return img
            
            
            


def shadeGouraughRev(verts_p,
                     verts_n,
                     verts_c,
                     bcoords,
                     cam_pos,
                     mat,
                     lights,
                     light_amb,
                     img):
    vertices_of_edge, x_limits_of_edge, y_limits_of_edge, sigmas_of_edges = compute_edge_limits(verts_p)
    x_min, x_max = int(np.amin(x_limits_of_edge)), int(np.amax(x_limits_of_edge))
    y_min, y_max = int(np.amin(y_limits_of_edge)), int(np.amax(y_limits_of_edge))
    
    # Active Elements of the triangle
    # Active Points: Points where we estimate the colro
    # Active Edges: Edges of the trinagles that are used for interpolating the color 
    
    # TODO fix this 
    n1,n2,n3 = verts_n(0),verts_n(1),verts_n(2)
    
    active_edges = np.array([False,False,False])
    active_points = np.zeros((3,2))
     # nodes-edges dict 
    nodes_on_edge = {0: [0,1],
                     1: [0,2],
                     3: [1,2]}
    
    
    c1_updated = verts_c[0]
    c2_updated = verts_c[1]
    c3_updated = verts_c[2]
    # Create Phong Illumination Objects for every Light Source :=: PointLightObject
    phongLightingSource = [None]*len(lights)
    for i,_ in enumerate(lights):
        phongLightingSource[i] = PhongIlluminationModel(mat.ka,mat.kd,mat.ks,mat.n_phong,lights[i].l_pos,lights[i].l_int)    
        c1_updated += phongLightingSource[i].light(bcoords,n1,verts_c[0],cam_pos,light_amb)
        c2_updated += phongLightingSource[i].light(bcoords,n2,verts_c[1],cam_pos,light_amb)
        c3_updated += phongLightingSource[i].light(bcoords,n3,verts_c[2],cam_pos,light_amb)   
    
    verts_c_updated = np.array([c1_updated,c2_updated,c3_updated])    
    """
        Scanline For Color Interpolation
    """
    # Apply scanline second step of interpolation for the active nodes
        
                
    for y in range (y_min,y_max):
        # every loop := every scanline update the active elements
        active_edges, active_points, updated_nodes = update_active_edges(y,vertices_of_edge,y_limits_of_edge,active_edges,active_points)
        updated_nodes =update_active_nodes(sigmas_of_edges,active_points,updated_nodes)
        # update the canvas := img variable
        img, active_points_color = color_interp(y,nodes_on_edge,x_limits_of_edge,y_limits_of_edge,sigmas_of_edges,active_edges,active_points,verts_c_updated,img)
        '''Scanline for Color Vectors'''
        x_left_bound , index_l = np.min(active_points[active_edges, 0]), np.argmin(active_points[active_edges, 0])
        x_right_bound, index_r = np.max(active_points[active_edges, 0]), np.argmax(active_points[active_edges, 0])
        c1 = active_points_color[active_edges][index_l]
        c2 = active_points_color[active_edges][index_r]
        cross_counter = 0  
        for x in range(x_min, x_max + 1):
            cross_counter += np.count_nonzero(x == np.around(active_points[active_edges, 0]))
            if cross_counter % 2 != 0 and int(np.around(x_left_bound)) != int(np.around(x_right_bound)):            
                img[x, y] = InterpolateVector(int(np.around(x_left_bound)), int(np.around(x_right_bound)), x, c1, c2,dim=2)
    return img
            
        