from DICTOL import *

acc = SRC_top('myYaleB', 15, 0.001)

acc = DLSI_top('myYaleB', 15, 13, 0.001, .01, False, True)

acc = COPAR_top('myYaleB', 15, 13, 5, 0.001, 0.01)

acc = FDDL_top('myYaleB', 15, 13, 0.001, 0.01, False, True)

acc = LRSDL_top('myYaleB', 15, 13, 5, 0.001, 0.01, .01, False, True)
