
import numpy as np 
from src.common_utils.render import renderObject,renderImageToFile
from src.common_utils.load_vertices import load_numpy_data
from src.common_utils.transformations import getRotmat
from src.common_utils.affine_transform import affine_transform


def main():
    image_Height, image_Width, canvas_Rows,canvas_Cols = 16, 16, 512, 512
    f = 70


    " Pipeline: load data from numpy encoded file, project 3D points to 2D and render the image <<captured>> "
    " 1. Load data from numpy encoded file "
    " 2. Project 3D points to 2D "
    " 3. Render the image <<captured>> "

    # 1. Load data from numpy encoded file
    p3d, faces, vcolors, u, ck, cu, cv,t1, t2, phi = load_numpy_data(filename='./GHW2-assets/h2.npy')
    # print("Loaded data from numpy encoded file: \n")
    # print("3D points: \n", p3d)
    # print("Faces: \n", faces)
    # print("Vertex colors: \n", vcolors)
    # print("Camera center: \n", ck)
    # print("Camera up: \n", cu)
    # print("Camera view: \n", cv)
    # print("Translation vector t1: \n", t1)
    # print("Translation vector t2: \n", t2)
    # print("Rotation angle phi: \n", phi)
    # print("Rotation axis u: \n", u)
    print("Starting with Loaded Image: \n")
    img1 = renderObject(p3d.T,faces,vcolors,image_Height,image_Width,canvas_Rows,canvas_Cols,f,cv,ck,cup=cu)
    renderImageToFile(img1 , 'img1',save=True)
    A = getRotmat(phi,u)
    verts3d_transformed1 = affine_transform(cp=p3d, RotationMatrix=A,translateVec=t1)
    #p3d,faces,vcolors,H,W,Rows,Columns,f,cv,cK,cup
    print("Rotate phi translate t1: \n")
    img2 = renderObject(verts3d_transformed1,faces,vcolors,image_Height,image_Width,canvas_Rows,canvas_Cols,f,cv,ck,cup=cu)
    renderImageToFile(img2 , 'img2',save=True)
    # Transformation 2 - only Rotation
    TranslateVec2 = np.zeros((3,1))
    print("Rotate phi: \n")
    verts3d_transformed2 = affine_transform(cp=p3d, RotationMatrix=A, translateVec=TranslateVec2)
    img3 =renderObject(verts3d_transformed2,faces,vcolors,image_Height,image_Width,canvas_Rows,canvas_Cols,f,cv,ck,cup=cu)
    renderImageToFile(img3 , 'img3',save=True)
    #Transformation 3 - Translation for t2 
    print("Translate t2: \n")
    verts3d_transformed3 = affine_transform(cp=p3d, RotationMatrix=np.eye(3),translateVec=t2)
    img4 = renderObject(verts3d_transformed3,faces,vcolors,image_Height,image_Width,canvas_Rows,canvas_Cols,f,cv,ck,cup=cu)
    renderImageToFile(img4 , 'img4',save=True)

    print("Done!")
    
if __name__ == "__main__":
    print("Rasterization Demo - Projections Transformations Initialized \n")
    main()