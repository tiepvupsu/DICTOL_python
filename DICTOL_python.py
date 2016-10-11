from matlab_syntax import * 
from utils import * 
from DLSI import * 
from ODL import * 
from COPAR import *

# acc = SRC_top('myARreduce', 7, 0.001)
# print acc 

acc = DLSI_top('myYaleB', 10, 10, 0.001, .01)
print acc 

acc = COPAR_top('myYaleB', 10, 10, 5, 0.001, 0.01)
print acc 