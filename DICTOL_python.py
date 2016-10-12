from matlab_syntax import * 
from utils import * 
from DLSI import * 
from ODL import * 
from COPAR import *
from LRSDL import *

# acc = SRC_top('myYaleB', 15, 0.001)
# print acc 

# acc = DLSI_top('myYaleB', 10, 8, 0.001, .01, False, True)
# # print acc 

acc = COPAR_top('myYaleB', 15, 12, 5, 0.001, 0.01)
# # print acc 
# acc = FDDL_top('myYaleB', 15, 15, 0.001, 0.01, False, True)
# # print acc 

acc = LRSDL_top('myYaleB', 15, 12, 5, 0.001, 0.01, .01, False, True)
# print acc 