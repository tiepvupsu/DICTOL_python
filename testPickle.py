import cPickle
import numpy as np 
import os 

import scipy.io as sio

def mat2pickle(filename):
	A = sio.loadmat(filename)
	pickle_fn = filename.replace('.mat', '.pickle')
	with open(r""+pickle_fn, "wb") as output_file:
	     cPickle.dump(A, output_file)

filename = os.path.join('data', 'myFlower103.mat')
mat2pickle(filename)