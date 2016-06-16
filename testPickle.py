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

# read 
pickle_fn = filename.replace('.mat', '.pickle')

def pickle_load(pick_fn):
    with open(pickle_fn, 'rb') as input_file:
        A = cPickle.load(input_file)
    # print A 
    return A 

pickle_load(pickle_fn) 


