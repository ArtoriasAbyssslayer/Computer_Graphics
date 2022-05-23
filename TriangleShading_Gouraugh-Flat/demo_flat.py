import numpy as np
import cv2 as cv 
import auxiliary_funcs as aux
import tqdm
import time
from scanline_util import TriangleFillingFunction 
# define canvas boundaries
M = 512
N = 512
# load vertices faces and arbitrary image elements from binary file
verts2d,vcolors,faces,depth  = aux.load_binary_data(filename = './GHW_assets/hw1.npy')

# set timmer to see the performance of the algorithm
toc = time.time()
print("Scanline algorithm running!:")
tff = TriangleFillingFunction()
img = tff.render(verts2d,faces,vcolors,depth,M,N,'flat')
tic = time.time()

# print the image to a file
aux.render_image_save_file(img,save=True, filename="flat_result")
print("Duration:",tic-toc)
