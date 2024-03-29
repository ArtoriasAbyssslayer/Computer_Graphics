import numpy as np
from src.common_utils.interpolate_color import interpolateVector as InterpolateVector 
from src.common_utils.scanlineUtilities import *

def initial_active_elements(active_edges, active_nodes, vertices_of_edge, y_limits_of_edge, sigma_of_edge):
    """Computes the initial active elements, at the start of the vertical scan

    Parameters
    ----------
    active_edges: the edges that are crossed by the vertical scan line
    active_nodes: the points at which the scan line intersects the active edges
    vertices_of_edge: the extremist vertices of every edge
    y_limits_of_edge: the min and max ordinate for every edge
    sigma_of_edge: the slope of every edge

    Returns
    -------
    active_edges : the updated active edges
    active_nodes : the updated active vertices
    is_invisible : indicator of whether or not, an invisible edge exists
    """
    y_min, _ = int(np.amin(y_limits_of_edge)), int(np.amax(y_limits_of_edge))
    is_invisible = False

   
    for i, y_limit in enumerate(y_limits_of_edge):
        if y_limit[0] == y_min:  # y-scan line meets new edge from the bottom
            if sigma_of_edge[i] == 0:  # lower horizontal line
                continue
            if np.isnan(sigma_of_edge[i]):  # it's an invisible line
                is_invisible = True
                continue
            active_edges[i] = True  # in other cases, it's an active line
            pos= np.argmin(vertices_of_edge[i, :, 1])
            active_nodes[i,0] = vertices_of_edge[i, pos, 0]
            active_nodes[i,1] = y_limits_of_edge[i, 0]
            
    return active_edges, active_nodes, is_invisible


def compute_edge_limits(verts2d):
    """Computes the max abscissa & ordinate values of every edge

    Notes
    -----
    Regarding the slope computation, there are three cases.
    1. positive/negative number means it's a line
    2. 0 means it's horizontal line
    3. float('inf') means it's a vertical line
    4. nan means it's a dot, not a line. So the triangle is a line (twisted inside z axis, not visible)

    Parameters
    ----------
    verts2d: the coordinates from the vertices of the triangle

    Returns
    -------
    edges_verts : the pair of vertices that make up every edge
    x_limits : the min and max abscissa of every edge
    y_limits : the min and max ordinate of every edge
    edges_sigma : the slope of every edge
    """
    edges_verts = np.array([[verts2d[0], verts2d[1]], [verts2d[0], verts2d[2]], [verts2d[1], verts2d[2]]])
    x_limits = np.array([np.min(edges_verts[:, :, 0], axis=1),
                         np.max(edges_verts[:, :, 0], axis=1)]).T
    y_limits = np.array([np.min(edges_verts[:, :, 1], axis=1),
                         np.max(edges_verts[:, :, 1], axis=1)]).T

    diff = np.array(edges_verts[:, 1] - edges_verts[:, 0])
    edges_sigma = np.array(diff[:, 1] / diff[:, 0])
    return edges_verts, x_limits, y_limits, edges_sigma


def update_active_edges(y, vertices_of_edge, y_limits_of_edge, sigma_of_edge, active_edges, active_nodes):
    """Updates the active edges, given a new scan line y

    Parameters
    ----------
    y : the current scan line
    vertices_of_edge : the pair of vertices that make up every edge
    y_limits_of_edge : the min and max ordinate of every edge
    sigma_of_edge : the slope of every edge
    active_edges : the updated active edges
    active_nodes : the updated active vertices

    Returns
    -------
    active_edges : the updated active edges
    active_nodes : the updated active vertices
    updated_nodes : indicates which vertex is already updated, so it is left intact by the 'update_active_nodes' method
    """
    updated_nodes = set()
    for i, y_limit in enumerate(y_limits_of_edge):
        if y_limit[0] == y:  # y-scan line meets new edge from the bottom
            if np.isnan(sigma_of_edge[i]):  # it's an invisible line
                continue
            active_edges[i] = True  # in other cases, it's an active line
            pos = np.argmin(vertices_of_edge[i, :, 1])
            active_nodes[i] = [vertices_of_edge[i, pos, 0], y_limits_of_edge[i, 0]]
            updated_nodes.add(i)
        if y_limit[1] == y:
            active_edges[i] = False

    return active_edges, active_nodes, updated_nodes


def update_active_nodes(sigma_of_edge, active_edges, active_nodes, updated_nodes):
    """Updates the active vertices, given a new scan line y

    Parameters
    ----------
    sigma_of_edge : the slope of every edge
    active_edges : the updated active edges
    active_nodes : the updated active vertices
    updated_nodes : which vertices were updated by the 'update_active_edges' routine

    Returns
    -------
    active_nodes : the updated active vertices x_updated += 1/m , y_updated += 1
    """
    for i, sigma in enumerate(sigma_of_edge):
        if active_edges[i] and sigma != 0 and i not in updated_nodes:
            active_nodes[i, 0] += 1 / sigma_of_edge[i]
            active_nodes[i, 1] += 1
    return active_nodes

# Color approx is applying the trilinear interpolation in steps in order to 
# solve the problem in 1dim
def color_interp(y, node_combination_on_edge, x_limits_of_edge, y_limits_of_edge, sigma_of_edge,
                  active_edges, active_nodes, vcolors, img):
    """Computes the color for the active vertices, and colors the horizontal edges

    Parameters
    ----------
    y : the current scan line
    node_combination_on_edge: the indices from the pair of vertices, that make up an edge
    x_limits_of_edge: the min and max abscissa of every edge
    y_limits_of_edge : the min and max ordinate of every edge
    sigma_of_edge : the slope of every edge
    active_edges : the updated active edges
    active_nodes : the updated active vertices
    vcolors: the RGB colors from the vertices of the triangle
    img: MxNx3 image matrix

    Returns
    -------
    img : the updated MxNx3 image matrix
    active_nodes_color : the RGB color, corresponding to every active vertex
    """
    active_nodes_color = np.zeros((3, 3))

    for i, point in enumerate(active_nodes):
        if active_edges[i]:
            # The edge coordinates of every active node. This is the edge i.
            x_edge = np.array(x_limits_of_edge[i])
            y_edge = np.array(y_limits_of_edge[i])
            node_pair = node_combination_on_edge[i]
            c1, c2 = vcolors[node_pair[0]], vcolors[node_pair[1]]
            # case of horizontal edge
            if sigma_of_edge[i] == 0:
                active_nodes_color[i] = InterpolateVector(x_edge[0], x_edge[1], active_nodes[i, 0], c1, c2,dim=2)
                for x in range(x_edge[0], x_edge[1]):
                    img[int(np.around(x)), int(np.around(y))] = InterpolateVector(x_edge[0], x_edge[1], x, c1, c2,dim=2)
            elif np.abs(sigma_of_edge[i]) == float('inf'):
                active_nodes_color[i] = InterpolateVector(y_edge[0], y_edge[1], y, c1, c2,dim=2)
                img[int(active_nodes[i, 0]), int(np.around(y))] = active_nodes_color[i]
            elif np.isnan(active_nodes[i,0]):
                continue
            else:
                active_nodes_color[i] = InterpolateVector(y_edge[0], y_edge[1], y, c1, c2,dim=2)
                img[int(active_nodes[i, 0]), int(np.around(y))] = active_nodes_color[i]

    return img, active_nodes_color