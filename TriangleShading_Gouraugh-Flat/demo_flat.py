import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
for i in tqdm(range(10)):
    # get the working directory path to __location__ variable
	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
	d = os.path.join(__location__,'./GHW_assets/hw1.npy')
	[verts2d,vcolors,faces,depth] = np.tolist(np.load(d,allow_pickle=True))
	# X = np.load(d)
	# np_load_old = X
	# #modify the default parameters of np.load()
	# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True,**k)
	# #call load_data with allow_pickle implicitly set to True
	#
	# (verts2d,vcolors,faces,depth) = imdb.load_data(num_words=10000)
	#
	# # restore np.load from future normal usage
	# np.load = np_load_old






	activeEdgesList = []
	activePointsSet = set()
	valuesToCheck = [col1,col2,col3,col4,col5,col6,col7];

	for coordinate in valuesToCheck:
		if coordinate in activeEdgesList:
			activePointsSet.add(coordinate)
			break

	# X = np.load(os.path.join(scriptpath,'hw1.npy'))
	# print(X)

	#
	# yk_min = y + 1; νεες πλευρες
	# yk_max = y εξαιρουμενες
