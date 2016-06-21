from matlab_syntax import * 
from utils import * 
from DLSI import * 
from ODL import * 
from COPAR import *

fn = os.path.join('data', 'tmp.pickle')
Vars = myload(fn)

# print Vars.keys()

D = Vars['D']
Y = Vars['Y']
# D_range_ext = Vars['D_range_ext']

lambda1 = 0.01
# DLSI_updateD(D, E, F, A, lambda1, opts)
C = 3 
N = 10
d = 30
k = 10
k0 = 10
c = 1

Y_range = np.array([0, 10, 20, 30])
D_range_ext = np.array([0, 10, 20, 30, 40])
DtD = np.dot(D.T, D)
DtY = np.dot(D.T, Y)
Yc = get_block_col(Y, c, Y_range)
Xc = zeros(D.shape[1], Yc.shape[1])
D_range = D_range_ext[: -1]

opts = Opts_COPAR(max_iter = 300, verbal = True, tol = 1e-8, \
    D_range_ext = D_range_ext, lambda1 = 0.01, eta = 0.1)

DCp1 = get_block_col(D, C, D_range_ext)
DCp1tDCp1 = np.dot(DCp1.T, DCp1)

k0 = D_range_ext[-1] - D_range_ext[-2]
if k0 > 0: 
    L = max_eig(DtD) + max_eig(DCp1tDCp1)
else:
    L = max_eig(DtD)


# DLSI_top_test()

# ODL_updateD(D, E, F, opts)

Xc = COPAR_updateXc(DtD, DCp1tDCp1, DtY,  Y_range, Xc, c, L, opts)
# print Xc