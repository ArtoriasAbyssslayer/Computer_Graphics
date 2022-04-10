import numpy as np
import cv2 as cv
import scanline_util
import auxiliary_funcs as aux
import tqdm
import time

# define canvas boundaries
M = 512
N = 512
# load vertices faces and arbitrary image elements from binary file


data,verts2d,vcolors,faces,depth = aux.load_binary_data(filename = './GHW_assets/hw1.npy')

# set timmer to see the performance of the algorithm
toc = time.time()
for i in tqdm.trange(100):
	print("Scanline algorithm running!:")
	tff = scanline_util.TriangleFillingFunction()
	img = tff.render(verts2d,faces,vcolors,depth,M,N,"gouraud")
	tic = time.time()

# print the image to a file
aux.render_image_save_file(img,save=True, filename="gouraud_result")
print("Duration:",tic-toc)
