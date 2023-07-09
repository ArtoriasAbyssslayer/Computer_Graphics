
# Internal Imports
from src.common_utils.shading import shadeFlat,shadeGouraud,shadePhong,shadeGouraudRev
from src.common_utils.rasterize import rasterize 
from src.common_utils.projections import cameraLookingAt
from src.common_utils.calculate_normals import calculate_normals as calculateNormals
from src.common_utils.calculate_Barycentrics import calculateBarycentricCoordinates as calculateBarycentrics
# Thrid Party PyPL
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
from tqdm import tqdm


"""
    1st Deliverable Code 
"""
shade_t = ['Flat','Gouraud','Phong','GouraughRev']
def render(verts2d,
           faces,
           vcolors,
           depth,
           m, n,
           shade_t):
    """Iterates over every triangle, from the farthest to the nearest, and calls the coloring method for each one separately.

    Parameters
    ----------
    verts2d : Lx2 matrix containing the coordinates of every vertex (L vertices)
    faces : Kx3 matrix containing the vertex indices of every triangle (K triangles)
    vcolors : Lx3 matrix containing the RGB color values of every vertex
    depth : Lx1 array containing the depth of every vertex in its initial, 3D scene
    shade_t : coloring strategy, with 'flat' and 'gouraud' indicating that every triangle should be filled with a
    color and have a gradual color changing effect respectively
    Returns
    -------
    canvas : MxNx3 image with colors
    """
    assert shade_t in ('flat', 'gouraud') and m >= 0 and n >= 0
    canvas = np.ones((m, n, 3))
    # depth of every triangle. depth[i] = depth of triangle i
    depth_tr = np.array(np.mean(depth[faces], axis=1))
    # order from the farthest triangle to the closest, depth-wise
    triangles_in_order = list(np.flip(np.argsort(depth_tr)))

    for tri in tqdm(triangles_in_order):
        triangle_verts = faces[tri]
        triangle_projected_verts = np.array(verts2d[triangle_verts])  # x,y of the 3 vertices of triangle t
        triangle_colors = np.array(vcolors[triangle_verts])  # color of the 3 vertices of triangle t
        if shade_t == 'flat':
            updated_canvas = shadeFlat(triangle_projected_verts, triangle_colors, canvas)
        elif shade_t == 'gouraud':
            updated_canvas = shadeGouraud(triangle_projected_verts, triangle_colors, canvas)
    normalized_image = (updated_canvas * 255).astype(np.uint8)
    cv.imshow('rendered image', normalized_image)
    cv.waitKey(1650)
    cv.destroyAllWindows()
    
    # fig, ax = plt.subplots()    
    # ax.imshow(normalized_image,cmap='viridis')
    # plt.show()
    # plt.pause(1)
    # plt.close()
    return updated_canvas
def render_with_illumination(verts2d,verts3d,
           faces,
           vcolors,
           depth,
           m, n,
           verts_normals,
           cam_pos,
           phongMaterial,
           pointLights,
           light_amb,
           canvas,
           LightingParam,
           shade_t):
    """Iterates over every triangle, from the farthest to the nearest, and calls the coloring method for each one separately.

    Parameters
    ----------
    verts2d : Lx2 matrix containing the coordinates of every vertex (L vertices)
    faces : Kx3 matrix containing the vertex indices of every triangle (K triangles)
    vcolors : Lx3 matrix containing the RGB color values of every vertex
    depth : Lx1 array containing the depth of every vertex in its initial, 3D scene
    shade_t : coloring strategy, with 'flat' and 'gouraud' indicating that every triangle should be filled with a
    color and have a gradual color changing effect respectively
    
    Additiona Parameters(Phong Reflection Model)
    --------------------
    verts_normals : Lx3 matrix containing the normals of every vertex
    bcoords: Kx3 matrix containing the barycentric coordinates of every triangle
    cam_pos: 3x1 vector containing the camera position
    phongMaterial: PhongMaterial object containing the phong reflection model coefficients
    pointLights: PointLight List of objects containing the light position and intensity for every light source
    light_amb: 3x1 vector containing the ambient light intensity
    
    
    Returns
    -------
    canvas : MxNx3 image with colors
    """
    assert shade_t in ('flat', 'gouraud','gouraudRev','phong') and m >= 0 and n >= 0
    # depth of every triangle. depth[i] = depth of triangle i
    depth_tr = np.array(np.mean(depth[faces], axis=0))
    # order from the farthest triangle to the closest, depth-wise
    triangles_in_order = list(np.flip(np.argsort(depth_tr)))
    for tri in tqdm(triangles_in_order ,desc="Processing triangles",total=len(triangles_in_order)):
        triangle_verts = faces[:,tri]
        triangle_projected_verts = np.array(verts2d[triangle_verts]) # x,y of the 3 vertices of triangle t
        tri_bcoords = np.mean(verts3d[:,triangle_verts], axis=1)
        triangle_colors = vcolors[:,triangle_verts]
        if shade_t == 'flat':
            updated_canvas = shadeFlat(triangle_projected_verts,verts_normals,triangle_colors,tri_bcoords,cam_pos,phongMaterial,pointLights,light_amb,canvas,LightingParam)
        elif shade_t == 'gouraud':
            updated_canvas = shadeGouraud(triangle_projected_verts,verts_normals,triangle_colors,tri_bcoords,cam_pos,phongMaterial,pointLights,light_amb,canvas,LightingParam)
        elif shade_t == 'gouraudRev':
            updated_canvas = shadeGouraudRev(triangle_projected_verts,verts_normals,triangle_colors,tri_bcoords,cam_pos,phongMaterial,pointLights,light_amb,canvas,LightingParam)
        elif shade_t == 'phong':
            updated_canvas = shadePhong(triangle_projected_verts,verts_normals,triangle_colors,tri_bcoords,cam_pos,phongMaterial,pointLights,light_amb,canvas,LightingParam)
            

    normalized_image = (updated_canvas * 255).astype(np.uint8)
    cv.imshow('rendered image', normalized_image)
    cv.waitKey(3000)
    cv.destroyAllWindows()
    
    # fig, ax = plt.subplots()    
    # ax.imshow(normalized_image,cmap='viridis')
    # plt.show()
    # plt.pause(1)
    # plt.close()
    return updated_canvas

"""
    Function RenderObject is the function that captures the object
    based on the second deliverable of the project modifications
"""

def renderObject(p3d,faces,vcolors,H,W,Rows,Columns,f,cv,cK,cup):
    """
        :param p3d: 3D coordinates of all vertices of the triangles
        :param faces: Indices of each triangle 3-verts tuple
        :param vcolors: The RGB values for each vertex
        :param H: Height of the image
        :param W: Width of the image
        :param Columns: Number of columns of the image
        :param f: focal length
        :param cv: camera loc WCS
        :param cK: no-homogenous camera lookat coordinates
        :param cup: camera up vector
    """
    I = np.zeros((H,W,3))
    # Compute the 2D coordinates of the vertices
    # and the depth of each tringle based on pinHole camera model - perspective projection
    verts2d, depths = cameraLookingAt(f,cv,cK,cup,p3d)
    rasterized_verts2d =  rasterize(verts2d,Rows=Rows,Columns=Columns,H=H,W=W)
    I = render(rasterized_verts2d,faces,vcolors,depths,512,512,shade_t='gouraud')
    return I


def displayTriangle(edge_verts):
    """ Displays an exact triangle
        based on the edge vertices
        that are supplied to the function
    """
    for i in range(3):
        Xs = list(edge_verts[i,0])
        Ys = list(edge_verts[i,1])
        plt.plot(Xs,Ys)


def renderImageToFile(canvas, filename, save=False):
    if save:
        imageio.imsave('./results/' + filename + '.png', (canvas * 255).astype(np.uint8))
    else:
        cv.imshow("Generated image", canvas)


"""
    3rd Deliverable Render function 
    Phong Shading,Illumination
"""
def render_object(shader,focal,eye,lookat,up,bg_color,
                  M,N,H,W,verts,vcolors,faces,mat,lights,light_amb,LightingParam):
    """
        Arguments:
        shader: binary value indicating the shading type (0: gouraud, 1: phong)
        focal: focal length of the camera that captures the object
        eye: camera location in WCS [3x1]
        lookat: point that the camera is looking at in WCS [3x1]
        up: up vector of the camera in WCS [3x1]
        bg_color: background color of the image
        M,N: dimension fo the produced image in pixels (MXN) pixels
        H,W: height and width of the camera sesnsor that captures the object
        verts: matrix of 3xNv vertices of the object
        vcolors: matrix 3xNv containing the color components of each vertesx
        faces: 3xN_T matrix that describes the triangles. k-th column has the increasing indices of the vertices of the k-th vertex
        mat: Is a material type PhongMaterial
        lights: List of light poistions in WCS
        light_amb: Ambient light intensity
        
    """
    
    img = bg_color
    normals = calculateNormals(verts,faces)
    Points,depths = cameraLookingAt(focal,eye,lookat,up,verts.T)
    rasterized_verts2d = rasterize(-Points,Rows=M,Columns=N,H=H,W=W)
    # case 1: gouraud shading Revised
    if shader == 0 :
        # render_with_illumination is the loop function on ordered depths which shades every triangle
        img = render_with_illumination(rasterized_verts2d,verts,faces,vcolors,depths,M,N,normals,eye,mat,lights,light_amb,img,LightingParam,shade_t='gouraudRev')
    # case 2: phong shading    
    elif shader == 1 :
        img = render_with_illumination(rasterized_verts2d,verts,faces,vcolors,depths,M,N,normals,eye,mat,lights,light_amb,img,LightingParam,shade_t='phong')
    
    return img