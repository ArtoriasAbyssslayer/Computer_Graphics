import numpy as np


def ambient_light(k,I_a):
	return np.matmul(k,I_a)
	 