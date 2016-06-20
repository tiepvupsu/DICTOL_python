from matlab_syntax import * 
from utils import * 
from DLSI import * 
from ODL import * 

fn = os.path.join('data', 'tmp.pickle')
Vars = myload(fn)

# print Vars.keys()

D = Vars['D']
E = Vars['E']
F = Vars['F']
A = Vars['A']

lambda1 = 0.01
opts = Opts_DLSI(max_iter = 300, verbal = True, tol = 1e-8)
DLSI_updateD(D, E, F, A, lambda1, opts)

# ODL_updateD(D, E, F, opts)