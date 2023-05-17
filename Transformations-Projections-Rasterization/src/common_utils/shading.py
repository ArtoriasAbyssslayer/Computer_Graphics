import numpy as np
from src.common_utils.interpolate_color import interpolateVector
from src.common_utils.scanlineUtilities import scanlineUtil

"""
    shadeFlat shades a single triangle 
    with flat shading 
    Vertices : the vertices of the tTraceback (most recent call last):
  File "/home/harry/Desktop/ΓραφικήΜεΥπολογιστές-Εργασια2023-24/CGEngine-Python/TriangleFilling/Main_flat.py", line 5, in <module>
    from common_utils.render import render
  File "/home/harry/Desktop/ΓραφικήΜεΥπολογιστές-Εργασια2023-24/CGEngine-Python/TriangleFilling/common_utils/render.py", line 3, in <module>
    from common_utils.shading import shadeFlat,shadeGouraud
  File "/home/harry/Desktop/ΓραφικήΜεΥπολογιστές-Εργασια2023-24/CGEngine-Python/TriangleFilling/common_utils/shading.py", line 2, in <module>
    from interpolate_color import interpolateVectorriangle 3x2 array
    Vcolors : the colors of each vertex 3x3 array
    canvas : the original canvas to draw on 
"""


def shadeFlat(canvas: np.ndarray, vertices: np.ndarray, vcolors: np.ndarray) -> np.ndarray:
    # Determine active edges and x-intersections for each scanline
    active_edges, active_points, x_bounds_scanlines, y_bounds = scanlineUtil(vertices)

    for y in range (y_bounds[0][0],y_bounds[0][1]):
        
        # Loop through intersection points of all scanlines of the trinagl
        for i in range(0, len(x_bounds_scanlines)):
            if active_edges is not None:
                if active_edges.shape == (2,):
                    for index,active_point in enumerate(active_points):
                        if active_point == 0:
                            continue
            # Compute color of pixel by taking mean of vertex colors
            pixel_color = np.mean(vcolors, axis=0)
            # Update canvas color
            if int(x_bounds_scanlines[i]) < 512 and int(x_bounds_scanlines[i]) >= 0:
                canvas[(int)(x_bounds_scanlines[i]), y, :] = pixel_color
            # else:
    return canvas

"""
    shadeGouraud shade a single triangle
    with gouraud shading interpolating between the 
    Scanline
    Vertices : the vertices of the triangle 3x2 array
    Vcolors : the colors of each vertex 3x3 array
    canvas : the original canvas to draw on 
"""


def shadeGouraud(canvas: np.ndarray, vertices: np.ndarray, vcolors: np.ndarray) -> np.ndarray:
    # Determine active edges and x-intersection for each scanline
    active_edges, active_points, x_bounds_scanlines, y_bounds = scanlineUtil(vertices)
   
    for y in range (y_bounds[0][0],y_bounds[0][1]):
        if active_edges.shape == 0:
            continue
        cross_counter = 0
        for i,x in enumerate(x_bounds_scanlines):
            x1 = int(x_bounds_scanlines[0])
            x2 = int(x_bounds_scanlines[-1])

            if x == None or x < 0:
                continue
            else:
                # Compute cross_counter
                cross_counter += np.count_nonzero(x == len(x_bounds_scanlines) - 1) + 1
                # Compute colors based on billinear interpolation between scanline active vertices
                if cross_counter % 2 != 0 and int(np.around(x1)) != int(np.around(x2)):
                    c1 = interpolateVector(active_points[1],active_edges[1],vcolors[0][:], vcolors[1][:],x,dim=2)
                    c2 = interpolateVector(active_points[1],active_edges[1],vcolors[1][:], vcolors[2][:],x,dim=2)
                    pixel_color = interpolateVector([int(np.around(x1)), y], [int(np.around(x2)), y],c1,c2, x, 2)
                    # Update canvas with pixel color
                    if x < 512:
                        canvas[int(x),y,:] = pixel_color
    return canvas