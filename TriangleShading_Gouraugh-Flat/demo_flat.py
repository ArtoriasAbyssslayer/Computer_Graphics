import numpy as np
import cv2 as cv
from scanline_util import TriangleFillingFunction as tff
import auxiliary_funcs as aux
import tqdm
import time 

# define canvas boundaries 
M = 512
N = 512
# load vertices faces and arbitrary image elements from binary file 


data,verts2d,vcolors,faces,depth  = aux.load_binary_data(filename = './GHW_assets/hw1.npy')

# set timmer to see the performance of the algorithm
toc = time.time()
for i in tqdm(range(int(966)):
    print("Scanline algorithm running!:")
    tff = TriangleFillingFunction()
    img = tff.render(verts2d,faces,vcolors,depth,M,N,"flat")
    tic = time.time()

# print the image to a file
aux.render_image_save_file(img,save=True, filename="flat_result")
print("Duration:" tic-toc)