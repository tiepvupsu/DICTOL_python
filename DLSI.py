from utils import *
from matlab_syntax import * 
from ODL import * 
from math import *

class Opts_DLSI:
    def __init__(self, lambda1 = 0.001, eta = 0.1, D_range= np.array([]), \
        max_iter = 100, verbose = False, tol = 1e-8):
        self.max_iter = max_iter
        self.verbose   = verbose
        self.tol      = tol
        self.lambda1  = lambda1
        self.eta      = eta
        self.D_range  = D_range


def DLSI_term(D, D_range):
    """
    cost =  DLSI_term(D, D_range):
    """ 
    A = erase_diagonal_blocks(np.dot(D.T, D), D_range, D_range)
    cost = normF2(A)
    return cost     

def DLSI_cost(Y, Y_range, D, D_range, X, opts):
    """
    * cost = DLSI_cost(Y, Y_range, D, D_range, X, lambda1, eta)
    * calculating DLSI cost function 
    * INPUT:
        - Y, D: np.array
        - Y_range, D_range: np.array 1d 
        - X: list of np arrays
    """ 
    C = Y_range.size - 1 
    cost = 0
    for c in xrange(C):
        Yc     = get_block_col(Y, c, Y_range)
        Xc     = X[c]
        Dc     = get_block_col(D, c, D_range)
        cost  += .5*normF2(Yc - np.dot(Dc, Xc)) + opts.lambda1*norm1(Xc)

    cost += 0.5*opts.eta *DLSI_term(D, D_range)
    return cost 
    
def DLSI_updateD(D, E, F, A, lambda1, opts):
    """
    function D = DLSI_updateD(D, E, F, A, lambda, opts)
    problem: `D = argmin_D -2trace(ED') + trace(FD'*D) + lambda *||A*D||F^2,` 
    subject to: `||d_i||_2^2 <= 1`
    where F is a positive semidefinite matrix
    ========= aproach: ADMM ==============================    
    rewrite: `[D, Z] = argmin -2trace(ED') + trace(FD'*D) + lambda ||A*Z||_F^2,` 
        subject to `D = Z; ||d_i||_2^2 <= 1`
    aproach 1: ADMM.
    1. D = -2trace(ED') + trace(FD'*D) + rho/2 ||D - Z + U||_F^2, 
        s.t. ||d_i||_2^2 <= 1
    2. Z = argmin lambda*||A*Z|| + rho/2||D - Z + U||_F^2
    3. U = U + D - Z
    solve 1: D = argmin -2trace(ED') + trace(FD'*D) + rho/2 ||D - W||_F^2 
                          with W = Z - U;
               = argmin -2trace((E - rho/2*W)*D') + 
                  trace((F + rho/2 * eye())*D'D)
    solve 2: derivetaive: 0 = 2A'AZ + rho (Z - V) with V = D + U 
    `Z = B*rhoV` with `B = (2*lambda*A'*A + rho I)^{-1}`
    `U = U + D - Z` 
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 5/11/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """
    def calc_cost(D):
        cost = -2*np.trace(np.dot(E, D.T)) + np.trace(np.dot(F, np.dot(D.T, D))) +\
            lambda1*normF2(np.dot(A, D))
        return cost 
    it    = 0
    rho   = 1.0
    Z_old = D.copy()
    U     = np.zeros_like(D)
    I_k   = eye(D.shape[1])
    X     = 2*lambda1/rho*A.T
    Y     = A.copy()
    B1    = np.dot(X, inv_IpXY(Y, X))

    # B1 = np.dot(X, LA.inv(eye(Y.shape[0]) + np.dot(Y, X)))
    tol = 1e-8
    optsD = Opts_DLSI(max_iter = 100)

    while it < opts.max_iter:
        it += 1 
        # update D 
        W  = Z_old - U 
        E2 = E + rho/2*W 
        F2 = F + rho/2*I_k 
        # D_old = D.copy() 
        D  = ODL_updateD(D, E2, F2, optsD)
        # print normF2(D - D_old)
        # update Z 
        V     = D + U
        Z_new = rho*(V - np.dot(B1, np.dot(Y, V)))
        e1    = normF2(D - Z_new)
        e2    = rho*normF2(Z_new - Z_old)
        if e1 < tol and e2 < tol:
            break 
        if opts.verbose:
            cost = calc_cost(D)
            print 'iter = %3d | costD = %5.4f | normF2(D - Z) = %5.4f | rho(Z_new - Z_old = %5.4f' \
                %(it, cost, e1, e2)
        # update U 
        U     = U + D - Z_new
        Z_old = Z_new.copy()

    return D 

def DLSI_updateD_test():
    d       = 300
    N       = 10
    k       = 5
    k2      = 495
    Y       = normc(np.random.rand(d, N))
    D       = normc(np.random.rand(d, k))
    X       = 1 * np.random.rand(k, N)
    E       = np.dot(Y, X.T)
    F       = np.dot(X, X.T)
    A       = normc(np.random.rand(k2, d))
    lambda1 = 0.01;
    opts    = Opts_DLSI(max_iter           = 300, verbose = True, lambda1 = 0.01)
    DLSI_updateD(D, E, F, A, lambda1, opts)

# DLSI_updateD_test()

def DLSI(Y, Y_range, opts, show_progress = False):
    """
     Syntax: `D, X = DLSI(Y, Y_range, opts)`
    * The main DLSI algorithm 
    * INPUT: 
      - `Y, Y_range`: training samples and their labels 
      - `opts`: a structure of parameters:
        + `lambda, eta`: `lambda` and `eta` in the cost function 
        + `max_iter`: maximum iterations. 
        + `D_range`: 
    * OUTPUT:
      - `rt`: total running time of the training process.   
    ============== Solutions ================
    * Cost functon:
        `[D, X, rt] = argmin_{D, X}(\sum 0.5*\|Y_c - D_c X_c\|_F^2) + 
              \lambda*norm1(X) + 0.5*eta * \sum_{i \neq c} \|D_i^T D_c\|_F^2`
    * updating `X`:
      `Xi = arg\min 0.5*||Y_c - D_c X_c\|_F^2 + \lambda \|X_c\|`
    * updating `D`:
      `Di = \arg\min \|Y_c - D_c X_c\|_F^2 + \eta \|D_{\c}^T D_c\|_F^2`
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 4/14/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """
    D_range = opts.D_range
    C = numel(Y_range) - 1
    D = zeros(Y.shape[0], D_range[-1])
    # ========= X should be stored in a list since different sizes =============
    X = []
    for c in xrange(C):
        n_rows = D_range[c+1] - D_range[c]
        n_cols = Y_range[c+1] - Y_range[c]
        X.append(zeros(n_rows, n_cols))

    lambda1 = opts.lambda1
    eta = opts.eta 
    # initialization 
    if opts.verbose:
        print 'Cost = %.5f' % DLSI_cost(Y, Y_range, D, D_range, X, opts)
        print 'Initiallizing... class:'

    opts_init = Opts_DLSI(lambda1 = lambda1, \
                                eta = opts.eta, \
                                max_iter = 50, \
                                verbose = False)
    print 'Initializing...'
    for c in range(C):
        if opts.verbose:
            print '%3d' %(c +1),
            if (c+1)%10 == 0:
                print '' # newline 
        Yc = get_block_col(Y, c, Y_range)
        Dc, X[c] = ODL(Yc, D_range[c+1] - D_range[c], opts.lambda1, opts_init)
        D[:, D_range[c]: D_range[c + 1]] = Dc 
        if not opts.verbose and show_progress:
            str0 = progress_str(c+1, C)
            sys.stdout.write("\r%s class: %d/%d" % (str0, c+1, C ))
            sys.stdout.flush()
    print '\n',
    if opts.verbose:
        print '\ncost_init = %.4f' %DLSI_cost(Y, Y_range, D, D_range, X, opts)

    it = 0 
    optsX = Opts_DLSI(lambda1 = opts.lambda1,\
                        max_iter = 300,\
                        verbose = False)
    optsD = Opts_DLSI(lambda1 = opts.lambda1,\
                        max_iter = 100,\
                        eta = opts.eta,\
                        verbose = False)
    t1 = time.time()
    ## MAIN algorithm
    while it < opts.max_iter:
        it += 1 
        # update X 
        for c in xrange(C):
            Yc   = get_block_col(Y, c, Y_range)
            Dc   = get_block_col(D, c, D_range)
            X[c] = lasso_fista(Yc, Dc, X[c], opts.lambda1, optsX)[0]
        if opts.verbose:
            print '\niter = %3d' %it, '/%3d' %opts.max_iter,
            costX = DLSI_cost(Y, Y_range, D, D_range, X, opts)
            print '| costX = %.4f' %costX,

        # update D 
        for c in xrange(C):
            # D_comi = D.copy()
            D_comc = np.delete(D, range(D_range[c], D_range[c+1]), axis = 1)
            Dc = get_block_col(D, c, D_range)
            Yc = get_block_col(Y, c, Y_range)
            E  = np.dot(Yc, X[c].T)
            F  = np.dot(X[c], X[c].T)
            A  = D_comc.T
            D[:, D_range[c]: D_range[c+1]] = DLSI_updateD(Dc, E, F, A, eta, optsD)
        t2 = time.time()
        if opts.verbose:
            costD = DLSI_cost(Y, Y_range, D, D_range, X, opts)
            print '                  costD = %.4f' %costD,
            t = (t2 - t1)*(opts.max_iter - it)/it 
            time_estimate(t)
        if t2 - t1 > 20*3600: # 20h
            break 
        if not opts.verbose and show_progress:
            str0 = progress_str(it, opts.max_iter, 50)
            sys.stdout.write("\r%s %.2f%%" % (str0, (it*100.0)/opts.max_iter ))
            sys.stdout.flush()
    print ''
    return (D, X)

def DLSI_test():
    d       = 300
    C       = 10
    N       = 10
    k       = 5
    Y       = normc(np.random.rand(d, N*C))
    Y_range = N*np.asarray(range(C+1))
    D_range = k*np.asarray(range(C+1))
    opts    = Opts_DLSI(max_iter = 100, D_range = D_range, lambda1 = 0.001, eta = 0.1, verbose = False)
    DLSI(Y, Y_range, opts)

# DLSI_test()

def DLSI_pred(Y, D, opts):
    """
    function pred = DLSI_pred(Y, D, opts)
    j = \arg\min_j R(y, Dj) with R(y,D) = 0.5*\|y - Dx\|_2^2 + lambda*\|x\|_1;
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 5/11/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """
    D_range = opts.D_range 
    C = numel(opts.D_range) - 1 
    E = zeros(C, Y.shape[1])
    optsX = Opts(max_iter = 300)
    for c in xrange(C):
        Dc = get_block_col(D, c, D_range)
        Xc = lasso_fista(Y, Dc, zeros(Dc.shape[1], Y.shape[1]), opts.lambda1, optsX)[0]
        R1 = Y - np.dot(Dc, Xc)
        E[c,:] = 0.5*(R1*R1).sum(axis = 0) + opts.lambda1*abs(Xc).sum(axis = 0)
    pred = np.argmin(E, axis = 0) + 1

    return pred 

def DLSI_top(dataset, n_c, k, alambda, eta, verbose = False, show_progress = True):
    """
    DLSI_top(dataset, n_c, k, alambda, eta)
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/19/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """
    print "=================DLSI================="
    print "Apply DLSI on " + dataset + " with parameters:"
    print "n_c: ", n_c, '\nk: ', k, '\nlambda: ',\
            alambda, '\neta: ', eta
    print '------------------'
    ## get data
    # print "\nPreparing training and test samples...",
    dataset, Y_train, Y_test, label_train, label_test = \
        train_test_split(dataset, n_c)

    C = np.unique(label_train).size 
    D_range     = k*np.arange(C+1)
    opts = Opts_DLSI(   max_iter = 100, \
                        lambda1  = alambda, \
                        D_range  = D_range,\
                        verbose  = verbose,\
                        eta      = eta)
    Y_range     = label_to_range(label_train)
    ## Train 
    print "Training ...",
    D, X = DLSI(Y_train, Y_range, opts, show_progress)
    print "...done training"
    ##
    print "Test...",
    pred = DLSI_pred(Y_test, D,  opts)
    acc = calc_acc(pred, label_test)
    # print acc 
    print 'Overall accuracy: %.2f %%' % (acc*100) 
    print ''
    # print "...done test"
    return acc 

def DLSI_top_test():
    DLSI_top('myYaleB', 15, 10, 0.001, .01)