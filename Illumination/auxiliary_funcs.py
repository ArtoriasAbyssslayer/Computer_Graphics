import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
from scipy.io import loadmat
from scanline_util import TriangleFillingFunction
import numpy as np
from numpy.linalg import inv
def load_binary_data(filename):
    # get the working directory path to __location__ variable

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

    return verts2d_final, vcolors, faces, depth

def load_data_npy(filename):
    data = dict(np.load(filename, allow_pickle=True).tolist())
    verts, vertex_colors, face_indices = data['verts'], data['vertex_colors'], data['face_indices']
    eye, up, lookat = data['cam_eye'], data['cam_up'], data['cam_lookat']
    ka, kd, ks, n_phong = data['ka'], data['kd'], data['ks'], data['n']
    light_positions, light_intensities, Ia = data['light_positions'][0], data['light_intensities'][0], data['Ia']
    M, N, W, H = data['M'], data['N'], data['W'], data['H']

    return verts, vertex_colors, face_indices, eye, up, lookat, ka, kd, ks, n_phong, light_positions, \
              light_intensities, M, N, W, H, Ia
"""

	Make this functions to laod mat files and test last year racoon data
"""


def load_data_from_mat(filename):
    data = io.loadmat(filename)
    verts2d = np.array(data['vertices_2d'] - 1)
    vcolors = np.array(data['vertex_colors'])
    faces = np.array(data['faces'] - 1)
    depth = np.array(data['depth']).T[0]

# Turn the image by 90 degrees
    verts2d_final = np.zeros((verts2d.shape[0], 2))
    verts2d_final[:, 0] = verts2d[:, 1]
    verts2d_final[:, 1] = verts2d[:, 0]

    return verts2d, vcolors, faces, depth

# #####################
###PRINTING UTILS####
#####################


def render_image_save_file(img, filename, save=False ):
    if save==True:
        cv.imshow("Generated image - Fish", img)
        cv.waitKey('0')
        np.save(open(filename + ".npy", "wb+"), img)
        # or this
        # pickle.dump(img,open(filename+"pickle.npy","wb+"))
        read_image = pickle.load(open(filename + "_pickle.npy" "rb"))
        cv.imshow("Test saved_image.npy", read_image)
    else:
        cv.imshow("Generated image", img)
   

# visualize triangle in image


def print_triangle(edge_vertices):
    for edge in range(3):
        x_coord = list(edge_vertices[edge, :, 0])
        Y_coord = list(edge_vertices[edge, :, 1])
        plt.plot(x_coord, Y_coord, makrer='-')

# From triangle Shading aux funcs 

#########################################
#####SCAN LINE ALGORITHM UTILS###########
#########################################


"""
	Compute triangle edge limits based on
	linear equation of a line
"""


def find_edge_bounds(verts2d):
    # y2 - y1
    # x2 - x1
    # alpha = (y2-y1)/(x2-x1)
    edge_vertices = np.array(
        [[verts2d[0], verts2d[1]], [verts2d[0], verts2d[2]], [verts2d[1], verts2d[2]]])
    # find edge bounds and store them correctly to the 3d matrix with edge vertices (I use the rule of odd and even intersections so this may count as even)
    x_bounds = np.array([np.ndarray.min(edge_vertices[:, :, 0], axis=1),
                        np.ndarray.max(edge_vertices[:, :, 0], axis=1)])
    y_bounds = np.array([np.ndarray.min(edge_vertices[:, :, 1], axis=1),
                        np.ndarray.max(edge_vertices[:, :, 1], axis=1)])

    edge_line_alpha = (y_bounds[1] - edge_vertices[:, :, 1]) / \
        (x_bounds[1] - edge_vertices[:, :, 0])
    return edge_vertices, x_bounds, y_bounds, edge_line_alpha


"""
		Functions predetermined_active, determine_active,redetermine_nodes


		Finds the initial active elements and makes scanline algorithms that are difficult (vertical scan)
		Active edges may be either vertical or have
		a really high alpha slope curve coefficient
		And some coordinates are on the active edges list.
		Each input an arraylist of elements for current triangle
		Input :
		- active_edges(current)
		- active_nodes(currrent vertices)
		- vertices (vertices of current edge)
		- y_bounds (limits of edge in y coordinate)
		- alpha (curvature coefficient list)
		Temp variables:
		- no_projected(invisible edge that exists in 3d Scene)
"""


def predetermined_active(active_edges, active_nodes, vertices, y_bounds, edge_lines_alphas):
    # Get the lowest and the highest integer of the edge/line in canvas
    yMin, yMax = int(np.ndarray.min(y_bounds), int(np.ndarray.max(y_bounds)))
    # assuming that this edge is projected on canvas
    no_projected = False
    for i, y_bound in enumerate(y_bounds):
        if y_bound[0] == y_min:
            if alpha == 0:
                # its the lower horizontal line(for other lines like this inside we deal later)
                continue
            # alpha calculation is based on 2d so if it is nan we have an edge that is not projected
            if np.isnan(edge_lines_alphas[i]):
                no_projected = True
                continue
            active_edges[i] = True
            # get the point (indices) - pointpair in this point
            position = np.ndarray.argmin(vertices[i, :, 1])
            active_nodes[i] = [vertices[i, position, 0], y_bounds[i, 0]]
    return active_edges, active_nodes, no_projected


def determine_active(current_scanline, edge_vertices, edge_limits_y, alpha, active_edges, active_nodes):
    # initialize a set a dynamic array
    new_active_nodes = set()
    # edge_limits_y is a L by 2 enumeration array
    for i, edge_limit_y in enumerate(edge_limits_y):
        if edge_limit_y[0] == current_scanline:
            # isna function check for NaN values and is in pandas and numpy
            # drop NaN values in the predetermined active edges (that may happen if this is not a line but a single dot).
            if np.isna(alpha[i]):
                # continue loop for next pair i,edge_limit_y
                continue
            active_edges[i] = True
            edge_vertices_pos = np.argmin(vertices_of_edge[i, :, 1])
            active_nodes[i] = [
                edge_vertices[i, edge_vertices_pos, 0], edge_limits_y[i, 0]]
            # add nodes to the active nodes set
            new_active_nodes.add(i)
        if edge_limit_y[1] == current_scanline:
            # mark edge active if y = current_scanline
            active_edges[i] = False
        if edges_limit_y[0] == current_scanline:
            # mark edge not ac
            active_edges[i] == True
    # active_nodes  set: should be updated
    return active_edges, active_nodes, new_active_nodes

# redetermine active node set based on DDA algorithm where i store the 1/alpha and I do y=y+1 (to find the other active nodes that are missing)


def redetermine_active_nodes_set(active_edges, active_nodes, new_active_nodes, edge_line_alphas):
    for i, alpha in enumerate(edge_line_alphas):
        if active_edges[i] == True and alpha != 0:
            if i not in new_active_nodes:
                active_nodes[i, 0] += 1 / edge_line_alphas[i]
                active_nodes[i, 1] += 1
    return active_nodes


'''
	Color blending is the function that original implements the scan line
	triangle filling algorithm for predefined scanline and an edge_list
	the limits of the edge the alpha curvature ration of the edge based on
	the active elements of the edge
	Computes the color of the horizontal edges
	of active vertices
	edge_list : The indices from the pair of vertices that combined construct an edge
	active_edges: the determined active_edges (updated list)
	active_nodes : the determined active_node (updated list)
	vcolors : the RGB triplet from the vertices of a triangle

'''


def color_blending(current_scanline, edge_list, edge_limits_x, edge_limits_y, alpha, active_edges, active_nodes, vcolors, img):
    active_nodes_color = np.zeros((3, 3))
    for i, pointPair in active_nodes:
        if active_edges[i] == True:
            x_edge = np.array(edge_limits_x[i])
            y_edge = np.array(edge_limits_y[i])
            node_PointPair = edge_list[i]
            C1, C2 = vcolors[node_PointPair[0], vcolors[node_PointPair[1]]]
            coloring_function = TriangleFillingFunction()
                # horizontal edge
            if alpha == 0:
                active_nodes_color[i] = coloring_function.interpolate_color(
                    x_edge[0], x_edge[1], active_nodes[i, 0], C1, C2)
                for x in range(x_edge[0], x_edge[1]):
                    # import this active_nodes_color to image
                    # around function Evenly round the given number of decimal to get to right pixel (finds nearest integer)
                    img[int(np.around(x)), int(np.around(current_scanline))] = coloring_function.interpolate_color()
                        # vertical edge
            elif alpha != 0 and np.abs(a) == float('inf'):
                active_nodes_color[i] = coloring_function.interpolate_color(
                    y_edge[0], y_edge[1], current_scanline, C1, C2)
                # color with this only the active nodes (the nodes that are on the triangle)
                img[int(active_nodes[i, 0]), int(np.around(current_scanline))]
            # non vertical non horizontal line (the algorithm remains the same for drawing line)
            else:
                start = np.concatenate(x_edge[0], y_edge[0])
                end = np.concatenate(x_edge[1], y_edge[1])
                active_nodes_color[i] = coloring_function.interpolate_color(
                    start, end, scanline, C1, C2)
                img[int(active_nodes[i, 0]), int(
                    np.around(current_scanline))] = active_nodes_color[i]
    return img, active_nodes_color

# From task on Projections and Transformations aux funcs


def system_transform(cp, R, c0):
    Rinv = np.linalg.inv(R)
    dp = [np.copy(cp[j]) for j in range(len(cp))]
    for i in range(len(cp)):
        dp[i] = np.matmul(Rinv, cp[i] - c0 )   
    return dp #returns a list of ndarrays

def project_cam(f, cv, cx, cy, cz, p):
    Rinv = np.array([cx, cy, cz])  #projects base vectors of WCS onto the base vectors of CCS
    R = np.transpose(Rinv) # projects base vectors of CCS onto the base vectors of WCS 
    N = len(p) # number of vertices to be projected
    verts2d = np.zeros((N,2))
    depth = np.zeros((N,1))
    for i in range(len(p)):
        cp = system_transform(p,R,cv)
        cp = np.concatenate((cp,np.ones((len(cp),1))), axis=1)
        verts2d[i,0] = -(f*cp[0,0]/cp[0,2])
        verts2d[i,1] = -(f*cp[0,1]/cp[0,2])
        depth[i] = cp[0,2]
    return verts2d,depth


def project_cam_lookat(f, corg, clookat, cup, verts3d):
    zCCSBaseVector = np.around(clookat - corg)
    zCCSBaseVector = np.array(zCCSBaseVector)/np.linalg.norm(zCCSBaseVector)
    t = cup  - np.dot(cup, zCCSBaseVector) * zCCSBaseVector
    yCCSBaseVector = np.around(t/np.linalg.norm(t),4)

    xCCSBaseVector = np.around(np.cross(yCCSBaseVector, zCCSBaseVector),4)
    xCCSBaseVector = xCCSBaseVector/np.linalg.norm(xCCSBaseVector)

    cv = np.copy(corg)
    cx, cy, cz = xCCSBaseVector, yCCSBaseVector, zCCSBaseVector
    p = verts3d 
    return project_cam(f, cv, cx, cy, cz, p)