import numpy as np




# Aux function that detects the pre-existing scanlines and
# returns the indices of the scanlines that are already present
# as long as which active_edges are correctly marked 
def already_active_components(edge_verts, active_verts, active_ybounds, edge_alpha):
    """
        Computes the initial active edges and active x bounds of the 
        already existing colored pixels.

        Args:
        active_edges : pass the active edges that are crossed by the horizontal scan line
        active_xbounds : the points at which the scanline intersects the active edges essentially the x_bounds
        edge_verts: the vertices of every edge
        active_ybounds: the min-max y coord of every edge passed
        edge_alpha: The slope coef of every edge
    """
    y_min, y_max = int(np.amin(active_ybounds)), int(np.amin(active_ybounds))
    are_active_edges = []
    is_invisible = False
    predetermined_active_edges = []
    for i, ybound in enumerate(active_ybounds):
        if ybound[0] == y_min:  # y-scanline meets new edge from bottom
            if edge_alpha[i] == 0:  # vertical line
                continue
            if np.isnan(edge_alpha).any():  # its a line with no slope aka invisible
                is_invisible = True
                continue
            are_active_edges.append(True)  # in other cases  that we have edges marke are_active_edges list true
            node_positions = np.argmin(edge_verts[i, :, 1])  # cases where we have only vertices that are active
            active_verts = (edge_verts[i, node_positions, 0], active_ybounds[i, 0])
            predetermined_active_edges = (edge_verts[i, node_positions, 0], active_ybounds[i, 0])
    return predetermined_active_edges, are_active_edges, active_verts, is_invisible


def isActive(y_bounds, edge_idx, currentY):
    if currentY >= y_bounds[edge_idx, 0] and currentY <= y_bounds[edge_idx, 1]:
        return True
    else:
        return False


def intersectionX(edge: np.ndarray, y):
    """
    Intersects gets the List of edges of the triangle 
    creates the line equations
    checks if the vertices are on the same scanline (y1 == y2)
    Then the function returns the intersection of the scanline which is the point with smaller x coord 
    If the y coords are different the intersection point is inside the edge
    So it reterns the  intersection point along the edge 
    
    """
    x1, y1 = edge[0, 0], edge[0, 1]
    x2, y2 = edge[1, 0], edge[1, 1]
    if y1 == y2:
        return min(x1, x2)
    else:
        m = (x2 - x1) / (y2 - y1)
        if m == 0:
            return y
        else:
            return (y - y1) / m + x1


def get_edge_bounds(verts2d):
    """
        Function that converts 2D vertices numpy array
        which is shape (#points, 2) -> numpoints, xy
        to edges_verts 
        and then compute edges bounds plus some other attributes of each edge
    """
    edge_vertices = np.array([[verts2d[0], verts2d[1]], [verts2d[0], verts2d[2]], [verts2d[1], verts2d[2]]])
    x_min = np.min(edge_vertices[:, :, 0], axis=1)
    x_max = np.max(edge_vertices[:, :, 0], axis=1)
    x_bounds = np.array(np.column_stack((x_min, x_max)), dtype=int)
    y_min = np.min(edge_vertices[:, :, 1], axis=1)
    y_max = np.max(edge_vertices[:, :, 1], axis=1)
    y_bounds = np.array(np.column_stack((y_min, y_max)), dtype=int)

    """
        x_bounds.shape = [#edges,xmin
                          #edges,xmax]
    """
    # compute DeltaX and DeltaY for each pair of vertices in edge_vertices pair array
    verts_deltas = np.array(edge_vertices[:, 1] - edge_vertices[:, 0])

    # compute for all edges the dy/dx 
    if (verts_deltas[:, 0].all() != 0 and verts_deltas[:, 1].all() != 0):
        edges_alpha = np.array(verts_deltas[:, 1] / verts_deltas[:, 0])
        edges_beta = np.array(verts2d[:, 1] - edges_alpha * verts2d[:, 0])
    else:
        edges_alpha = np.array(verts_deltas[:, 1] / verts_deltas[:, 0])
        edges_beta = np.array(verts_deltas[:, 1])

    return edge_vertices, x_bounds, y_bounds, edges_alpha, edges_beta


# update the active verts based on bresenham line slope
def updateActiveVerts(active_edges, active_verts, updated_verts, edge_alpha):
    """
        active_verts: Vertices that are already active elements

        Retyrbs tge yodated
    """
    for i, edge_alpha in enumerate(edge_alpha):
        if active_edges.any() and edge_alpha != 0 or i not in updated_verts:
            active_verts = (1 / edge_alpha, ++1)
        return active_verts


def updateActiveEdges(active_edges, edge_verts, y, edge_alpha, y_bounds):
    """
        active_edges = [edge_index, edge_verts]
    """
    updated_points = []
    updated_active_bounds = []
    is_active = []
    for i, y_bound in enumerate(y_bounds):
        if y_bound[0] == y:  # scanline meets the new edge from the bottom
            if np.isnan(edge_alpha).any():
                continue
            is_active.append(True)
            pos = np.argmin(edge_verts[i, :, 1])
            updated_active_bounds = [edge_verts[i, pos, 0], y_bounds[i, 0]]
            updated_points.append(updated_active_bounds)
        if y_bound[1] == y:
            is_active.append(False)
            updated_active_bounds.append(active_edges)
    return np.array(active_edges), updated_active_bounds, updated_points


def scanlineUtil(vertices: np.ndarray):
    """
    scanlineUtil returns the points of the canvas that are 
    inside the triangles
    Essentially the points of the scanline that are inside the 
    triangle bounds
    """
    # print(vertices)
    edges_verts, x_bounds, y_bounds, edge_alpha, _ = get_edge_bounds(vertices)
    # render.displayTriangle(edges_verts)
    y_min = np.min(y_bounds)
    y_max = np.max(y_bounds)
    active_edges = np.zeros((y_max - y_min, 2))
    active_verts = np.zeros((y_max - y_min, 2))
    active_verts.reshape((y_max - y_min, 2))
    x_bounds_scanlines = np.zeros((y_max - y_min,))
    updated_verts = np.zeros((y_max - y_min))
    for y in range(y_min, y_max):
        # print("Y max: ", y_max)
        # print("Y min: ", y_min)
        # print("edges_verts shape: ",edges_verts.shape)
        # print("active_verts shape",active_verts.shape)
        active_edges, active_verts, actives_edges_list, is_invisible = already_active_components(edges_verts,
                                                                                                 active_verts,
                                                                                                 y_bounds, edge_alpha)

        for edge_idx in range(len(edges_verts)):
            if isActive(y_bounds, edge_idx, y):
                if is_invisible:
                    break
                if len(active_edges) == 0:
                    continue
                # compute the intersection point

                # Triangle case
                if edges_verts[edge_idx, 0, 1] >= x_bounds[edge_idx, 0] and edges_verts[edge_idx, 0, 0] < x_bounds[
                    edge_idx, 1] or edges_verts[edge_idx, 1, 0] > x_bounds[edge_idx, 0] and edges_verts[
                    edge_idx, 1, 1] < x_bounds[
                    edge_idx, 1] or edges_verts[edge_idx, 0, 0] > x_bounds[edge_idx, 0] and edges_verts[
                    edge_idx, 1, 1] < x_bounds[
                    edge_idx, 1] or edges_verts[edge_idx, 1, 1] >= x_bounds[edge_idx, 1] and edges_verts[
                    edge_idx, 1, 0] < x_bounds[
                    edge_idx, 0] and edges_verts[edge_idx, 1, 1] == x_bounds[edge_idx, 1] and edges_verts[
                    edge_idx, 0, 0] == x_bounds[
                    edge_idx, 0] and edges_verts[edge_idx, 1, 0] == x_bounds[edge_idx, 0] and edges_verts[
                    edge_idx, 1, 1] == x_bounds[
                    edge_idx, 1] and edges_verts[edge_idx, 1, 0] == edges_verts[edge_idx, 0, 1] and edges_verts[
                    edge_idx, 0, 0] == edges_verts[edge_idx, 1, 1]:
                    x_intersect = intersectionX(edges_verts[edge_idx], y)
                    x_bounds_scanlines[y - y_min] = x_intersect

        active_edges, active_verts, updated_verts = updateActiveEdges(active_edges,
                                                                      edges_verts,
                                                                      y - y_min,
                                                                      edge_alpha,
                                                                      y_bounds)
        updated_verts = updateActiveVerts(active_edges,
                                          active_verts,
                                          updated_verts,
                                          edge_alpha)

    return active_edges, updated_verts, x_bounds_scanlines, y_bounds