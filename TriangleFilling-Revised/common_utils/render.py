import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imageio
from common_utils.shading import shadeFlat,shadeGouraud
from tqdm import tqdm
shade_t = ['Flat','Gouraud']

# Updates the canvas  with the FlatShadedTriangles
def renderFlatShade(verts2d,vcolors,canvas):
    updatedcanvas = shadeFlat(canvas, verts2d, vcolors)
    return updatedcanvas
    
   
def renderGouraudShade(verts2d,vcolors,canvas):
    updatedcanvas = shadeGouraud(canvas, verts2d, vcolors)
    return updatedcanvas
def render(verts2d, faces, vcolors, depths, shade_t):
   
    # canvas size 
    m = 512
    n = 512

    assert shade_t in ('flat', 'gouraud') and m >= 0 and n >= 0 


    # create canvas with white background
    canvas = np.ones((m,n,3))
    # depth of every triangle, depth[i] =  depth of triangle i 
    tri_depth = np.array(np.mean(depths[faces], axis=1))
    # order the K triangle by depth 
    tri_depth_ordered = list(np.flip(np.argsort(tri_depth)))

    # main render loop for each shading method 
    for tri in tqdm(tri_depth_ordered,desc='Processing triangles'):
        triangle_verts = faces[tri]
        triangle_projected_verts = np.array(verts2d[triangle_verts]) # x,y of the 3 vertices of triangle t
        triangle_colors = np.array(vcolors[triangle_verts])
        # choose shader
        if shade_t == 'flat':
            updated_canvas = renderFlatShade(verts2d=triangle_projected_verts,vcolors=triangle_colors,canvas=canvas)

        elif shade_t == 'gouraud':
            updated_canvas = renderGouraudShade(verts2d=triangle_projected_verts, vcolors=triangle_colors, canvas=canvas)

    updated_canvas = np.float32(updated_canvas)
    updated_canvas = cv.cvtColor(updated_canvas, cv.COLOR_BGR2RGB)
    cv.imshow('rendered image', updated_canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()
    renderImageToFile(updated_canvas, 'updated_canvas',save=True)
    plt.show()
    return updated_canvas



def displayTriangle(edge_verts):
    """ Displays an exact triangle
        based on the edge vertices
        that are supplied to the function
    """

    for i in range(3):
        Xs = list(edge_verts[i,0])
        Ys = list(edge_verts[i,1])
        plt.plot(Xs,Ys)


def renderImageToFile(img, filename, save=False):
    if save:
        imageio.imsave('./results/' + filename + '.png', (img * 255).astype(np.uint8))
    else:
        cv.imshow("Generated image", img)
