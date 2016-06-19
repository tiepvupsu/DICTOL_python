from utils import *
from ODL import *
#################### FDDL ##########################
def FDDL_fidelity(Y, Y_range, D, D_range, X):
    """
    * Syntax: cost = FDDL_fidelity(Y, Y_range, D, D_range, X)
    * Calculating the fidelity term in FDDL[[4]](#fn_fdd):
    * $\sum_{c=1}^C \Big(\|Y_c - D_cX^c_c\|_F^2 + 
        \sum_{i \neq c} \|D_c X^c_i\|_F^2\Big)$
    """
    cost = 0 
    C = Y_range.size - 1 
    for c in xrange(C):
        Yc   = get_block_col(Y, c, Y_range)
        Dc   = get_block_col(D, c, D_range)
        Xc   = get_block_row(X, c, D_range)
        Xcc  = get_block_col(Xc, c, Y_range)
        cost += normF2(Yc - np.dot(Dc, Xcc))
        for i in xrange(C):
            if i == c:
                continue 
            Xci = get_block_col(Xc, i, Y_range)
            cost += normF2(np.dot(Dc, Xci))
    return cost 

def FDDL_discriminative(X, Y_range):
    """
    * Syntax: cost = FDDL_discriminative(X, Y_range)
    * calculating the discriminative term in FDDL[[4]](#fn_fdd):
    * $\|X\|_F^2 + \sum_{c=1}^C (\|Xc - Mc\|_F^2 - \|Mc - M\|_F^2) $
    """
    cost = normF2(X) 
    C = Y_range.size - 1 
    m = np.mean(X, axis = 1)
    for c in xrange(C):
        Xc   = get_block_col(X, c, Y_range)
        Mc   = build_mean_matrix(Xc)
        cost += normF2(Xc - Mc)
        M    = repmat(m, 1, Xc.shape[1])
        # import pdb; pdb.set_trace()  # breakpoint d8aa75a6 //
        cost -= normF2(Mc - M)
    return cost 

def FDDL_cost(Y, Y_range, D, D_range, X, opts):
    cost = 0.5*normF2(Y - np.dot(D, X)) + opts.lambda1*norm1(X) + \
           0.5*FDDL_fidelity(Y, Y_range, D, D_range, X) + \
           0.5*opts.lambda2*FDDL_discriminative(X, Y_range)
    return cost 

def FDDL_updateX(Y, Y_range, D, D_range, X, opts):
    pass 

def FDDL_updateD(Y, Y_range, D, D_range, X, opts):
    pass 

def FDDL_updateD_fast(Y, Y_range, D, D_range, X, opts):
    """
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/22/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """ 
    F = buildMhat(np.dot(X, X.T), D_range, D_range)
    E = np.dot(Y, buildMhat(X.T, Y_range, D_range))
    # print 'diff E, E3', LA.norm(E - E3)
    optsD = Opts(max_iter = 200, tol = 1e-8)
    return ODL_updateD(D, E, F, optsD)

def FDDL_pred(Y, Y_label, D, D_range, M, opts):
    pass 