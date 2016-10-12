from utils import *
from matlab_syntax import * 
from ODL import * 
from math import *
from DLSI import * 
import time 

class Opts_COPAR:
    def __init__(self, lambda1 = 0, eta = 0,verbose = False, max_iter = 100, tol = 1e-8, \
        D_range_ext = np.array([0]),  check_grad = False):
        self.verbose      = verbose
        self.max_iter    = max_iter
        self.tol         = tol
        self.D_range_ext = D_range_ext
        self.lambda1     = lambda1
        self.eta         = eta
        self.check_grad  = check_grad

def COPAR_cost(Y, Y_range, D, D_range_ext, X, opts):
    """
    cost = COPAR_cost(Y, Y_range, D, D_range_ext, X, opts):
    Calculating cost function of COPAR with parameters lambda and eta are 
    stored in `opts.lambda` and `opts.rho`.
    `f(D, X) = 0.5*sum_{c=1}^C 05*||Y - DX||_F^2 + 
                  sum_{c=1}^C ( ||Y_c - D_Cp1 X^Cp1_c - D_c X_c^c||F^2 + 
              sum_{i != c}||X^i_c||_F^2) + lambda*||X||_1 + 
              0.5*eta*sum_{i \neq c}||Di^T*Dc||_F^2`
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 5/11/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------   
    """
    C          = numel(Y_range) - 1
    eta        = opts.eta
    alambda    = opts.lambda1
    cost       = alambda*norm1(X)
    cost1      = normF2(Y - np.dot(D, X))
    DCp1       = get_block_col(D, C, D_range_ext)
    DCp1_range = range(D_range_ext[-2], D_range_ext[-1])

    for c in xrange(C):
        Dc    = get_block_col(D  , c , D_range_ext)
        Yc    = get_block_col(Y  , c , Y_range)
        Xc    = get_block_col(X  , c , Y_range)
        Xcc   = get_block_row(Xc , c , D_range_ext)
        XCp1c = get_block_row(Xc , C , D_range_ext)

        cost1 = cost1 + normF2(Yc - np.dot(Dc, Xcc) - np.dot(DCp1, XCp1c))
        # Xc(union(Dc_range, DCp1_range),:) = []
        XX = Xc[: D_range_ext[-2], :] # remove X_{C+1}
        XX = np.delete(XX, range(D_range_ext[c], D_range_ext[c+1]) , axis = 0 );
        cost1 = cost1 + normF2(XX)
    cost = cost + cost1 + .5*eta*DLSI_term(D, D_range_ext)
    return cost 

def COPAR_init(Y, Y_range, opts, show_progress = False):
    C           = numel(Y_range) - 1
    D_range_ext = opts.D_range_ext
    D_range     = D_range_ext[:-1]
    k0          = D_range_ext[-1] - D_range_ext[-2]
    D           = zeros(Y.shape[0], D_range_ext[-1])
    X           = zeros(D_range_ext[-1], Y_range[-1])

    if opts.verbose:
        print 'Initiallizing class:...'
    for c in xrange(C):
        if opts.verbose:
            print '%3d' %(c + 1), 
            if (c+1) % 10 == 0:
                print ''
        Yc      = get_block_col(Y, c, Y_range)
        Dc, Xcc = ODL(Yc, D_range[c+1] - D_range[c], opts.lambda1, opts)
        D[:, D_range[c]: D_range[c+1]] = Dc 
        X[D_range[c]: D_range[c+1], Y_range[c] : Y_range[c+1]] = Xcc 
        ## progress bar 
        if not opts.verbose and show_progress:
            str0 = progress_str(c+1, C)
            sys.stdout.write("\r%s class: %d/%d" % (str0, c+1, C ))
            sys.stdout.flush()
    # if opts.verbose:
    if not opts.verbose and show_progress:
        print '\n',
    if k0 > 0:
        DCp1, XCp1 = ODL(Y, k0, opts.lambda1, opts)
        D[:, D_range_ext[-2]: D_range_ext[-1]] = DCp1 
        X[D_range_ext[-2]: D_range_ext[-1], :] = XCp1 
    return (D, X)


def COPAR_updateXc(DtD, DCp1tDCp1, DtY,  Y_range, Xc, c, L, opts):
    """
    function Xc = COPAR_updateXc(DtD, DtY,  Y_range, Xc, c, L, opts) 
    * Update Xc in COPAR (page 189-190 COPAR)
    see COPAR paper: 
    http://www.cs.zju.edu.cn/people/wangdh/papers/draft_ECCV12_particularity.pdf
     cost = normF2(Yc - D*Xc) + normF2(Yc - DcXcc - DCp1*XCp1c) + 
              sum_{i \neq c, 1 \leq i \leq C} normF2(Xic);
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 5/12/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """
    C = numel(Y_range) - 1 
    D_range_ext = opts.D_range_ext
    lambda1 = opts.lambda1/2
    
    ##
    def calc_f(Xc): # used in test mode only 
        cost   = normF2(Yc - np.dot(D, Xc))
        Xcc    = get_block_row(Xc, c, D_range_ext)
        XCp1c  = get_block_row(Xc, C, D_range_ext)
        Dc     = get_block_col(D, c, D_range_ext)
        DCp1   = get_block_col(D, C, D_range_ext)
        cost  += normF2(Yc - np.dot(Dc, Xcc) - np.dot(DCp1, XCp1c))
        # the following for loop could be optimized. However, because this function
        # is insignificant, it is used in the test mode only, I'll optimize it 
        # in a beautiful day
        for i in xrange(C):
            if i == c:
                continue 
            Xic = get_block_row(Xc, i, D_range_ext)
            cost += normF2(Xic)
        cost *= .5
        return cost 
    ## 
    def calc_F(Xc):
        return calc_f(Xc) + lambda1*norm1(Xc)
    ## For gradient function 
    DctDc       = get_block(DtD, c, c, D_range_ext, D_range_ext)
    DCp1tDc     = get_block(DtD, C, c, D_range_ext, D_range_ext)
    DctDCp1     = DCp1tDc.T
    # range_c   = D_range_ext(c)+1: D_range_ext(c+1)
    # range_Cp1 = D_range_ext(C+1) + 1: D_range_ext(C+2)
    DtYc        = get_block_col(DtY, c, Y_range)
    DtYc2       = DtYc.copy()
    DtYc2[D_range_ext[c]: D_range_ext[c+1],:] = \
            2*DtYc[D_range_ext[c]: D_range_ext[c+1],:]
    DtYc2[D_range_ext[-2]:D_range_ext[-1],:] = \
            2*DtYc[D_range_ext[-2]:D_range_ext[-1], :]
    k0 = D_range_ext[-1] - D_range_ext[-2]
    def grad(Xc0):
        Xc   = Xc0.copy()
        g0    = np.dot(DtD, Xc)
        Xcc   = get_block_row(Xc, c, D_range_ext)
        XCp1c = get_block_row(Xc, C, D_range_ext)
        if k0 > 0:
            Xc[D_range_ext[c]: D_range_ext[c+1], :] = \
                np.dot(DctDc, Xcc) + np.dot(DctDCp1, XCp1c)
            Xc[D_range_ext[-2]: D_range_ext[-1], :] = \
                np.dot(DCp1tDCp1, XCp1c) + np.dot(DCp1tDc, Xcc)
        else:
            Xc[D_range_ext[c]: D_range_ext[c+1], :] = np.dot(DctDc, Xcc)
        g = g0 + Xc - DtYc2
        return g 
    ## check_grad 
    # grad(Xc)
    if opts.check_grad:
        if not check_grad(calc_f, grad, np.random.rand(Xc.shape[0], Xc.shape[1])):
            print 'Check grad again before going further!!'
            return 
    ## Main FISTA 
    if k0 > 0:
        L2 = L + max_eig(DCp1tDCp1) + 2
    else:
        L2 = L + 2 

    opts_fista = Opts_COPAR(max_iter = 300, tol = 1e-8, verbose = False)
    Xc = fista(grad, Xc, L2, lambda1, opts_fista, calc_F)[0]
    return Xc


def COPAR_updateX(Y, Y_range, D, X, opts):
    """
    function X = COPAR_updateX(Y, Y_range, D, X, opts)
    updating X in COPAR. 
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 5/11/2016
            (http://www.personal.psu.edu/thv102/)
    ----------------------------------------------- 
    """
    C = numel(Y_range) - 1 
    DtD = np.dot(D.T, D)
    DtY = np.dot(D.T, Y)
    DCp1 = get_block_col(D, C, opts.D_range_ext)
    DCp1tDCp1 = np.dot(DCp1.T, DCp1)
    D_range_ext = opts.D_range_ext
    k0 = D_range_ext[-1] - D_range_ext[-2]
    if k0 > 0 :
        L = max_eig(DtD) + max_eig(DCp1tDCp1)
    else:
        L = max_eig(DtD)

    optsX = Opts_COPAR(verbose = False, max_iter = 100, lambda1 = opts.lambda1, \
                        D_range_ext = opts.D_range_ext)
    ##
    for c in xrange(C):
        Xc = get_block_col(X, c, Y_range)

        X[:, Y_range[c]: Y_range[c+1]] = COPAR_updateXc(DtD, DCp1tDCp1, DtY, \
            Y_range, Xc, c, L, optsX)
        if opts.verbose:
            costXc = COPAR_cost(Y, Y_range, D, D_range_ext, X, opts)
            print 'class = %3d | costXc = %.4f' %(c, costXc)

    return X 


def COPAR_updateD(Y, Y_range, D, X, opts):
    C = numel(Y_range) - 1 
    D_range_ext = opts.D_range_ext 
    DCp1 = get_block_col(D, C, D_range_ext)
    optsD = Opts_COPAR(max_iter = 100, verbose = False,\
                        eta = opts.eta, D_range_ext = opts.D_range_ext)
    Yhat = np.zeros_like(Y)
    ## update Dc 
    for c in xrange(C):
        # Dc = arg\min_Dc \|Ychat - Dc*Xcc\|_F^2 + \|Ycbar - Dc*Xcc\| + 2*eta\|A*Dc\|_F^2
        #  = \arg\min_Dc \| [Ychat Ycbar] - Dc*[Xcc Xcc]\|_F^2 + 2*eta\|A*Dc\|_F^2
        #  and solved using DLSI_updateD
        Dc_range = range(D_range_ext[c], D_range_ext[c+1])
        Yc_range = range(Y_range[c], Y_range[c+1])
        Yc       = Y[:, Yc_range]
        Dc       = D[:, Dc_range]
        Xc       = X[:, Yc_range]
        Xcc      = Xc[Dc_range, :]
        XCp1c    = get_block_row(Xc, C, D_range_ext)
        Ychat    = Yc - np.dot(D, Xc) + np.dot(Dc, Xcc)
        Ycbar    = Yc - np.dot(DCp1, XCp1c)
        E        = np.dot(Ychat + Ycbar, Xcc.T)
        F        = 2*np.dot(Xcc, Xcc.T)
        A        = D.copy()
        A        = np.delete(A, Dc_range, axis = 1)
        D[:, Dc_range] = DLSI_updateD(Dc, E, F, A.T, opts.eta, optsD)
        Yhat[:, Yc_range] = Yc - np.dot(D[:, Dc_range], Xcc)
        # print COPAR_cost(Y, Y_range, D, opts.D_range_ext, X, opts)
    ## DCp1 
    XCp1 = get_block_row(X, C, D_range_ext)
    Ybar = Y - np.dot(D[:, : D_range_ext[-2]], X[: D_range_ext[-2], :])
    E = np.dot(Ybar + Yhat, XCp1.T)
    F = 2*np.dot(XCp1, XCp1.T)
    A = D[:, : D_range_ext[-2]]
    DCp1_range = range(D_range_ext[-2], D_range_ext[-1])
    D[:, DCp1_range] = DLSI_updateD(D[:, DCp1_range], E, F, A.T, opts.eta, optsD)

    return D 

def COPAR(Y, Y_range, opts, show_progress = False):
    C = numel(Y_range) - 1
    D_range_ext = opts.D_range_ext
    D_range = D_range_ext[: -1]
    ## INIT 
    opts_init = Opts_COPAR(max_iter = 30, verbose = False, lambda1 = opts.lambda1,\
                eta = opts.eta, D_range_ext = opts.D_range_ext)
    D, X = COPAR_init(Y, Y_range, opts_init, show_progress)
    ## opts for X and D 
    optsX = Opts_COPAR(max_iter = 300, verbose = False, lambda1 = opts.lambda1, \
                        D_range_ext = opts.D_range_ext)
    optsD = Opts_COPAR(max_iter = 100, verbose = False, lambda1 = opts.lambda1, \
                        eta = opts.eta, D_range_ext = opts.D_range_ext)
    it = 0 
    t1 = time.time()
    print "Main Algorithm..."
    ## MAIN algorithm
    while it < opts.max_iter:
        it += 1
        ## update X 
        if opts.verbose:
            print 'iter = %3d/%3d | updating X ...' %(it, opts.max_iter),
        X = COPAR_updateX(Y, Y_range, D, X, optsX)
        t2 = time.time()
        if (t2 - t1) > 20*3600:
            break 
        if opts.verbose:
             costX = COPAR_cost(Y, Y_range, D, D_range_ext, X, opts)
             print 'costX = %5.3f' %costX 
        ## update D 
        if opts.verbose:
            print '                 updating D ...',
        D = COPAR_updateD(Y, Y_range, D, X, optsD); # and DCp1 
        t2 = time.time()
        if opts.verbose:
            costD = COPAR_cost(Y, Y_range, D, D_range_ext, X, opts)
            print 'costD = %5.3f' %costD,   
            t = (t2 - t1)*(opts.max_iter - it)/it 
            time_estimate(t)    
            if abs(costX - costD) < 1e-4:
                break 
        if (t2 - t1) > 20*3600:
            break 
        if not opts.verbose and show_progress:
            str0 = progress_str(it, opts.max_iter, 50)
            sys.stdout.write("\r%s %.2f%%" % (str0, (it*100.0)/opts.max_iter ))
            sys.stdout.flush()
            
    if not opts.verbose and show_progress:
        print ''
    return (D, X)

def COPAR_test():
    C = 20
    N = 7 
    d = 30
    k0 = 10
    k = 7

    D_range = k*np.arange(C+1)
    D_range_ext = np.hstack((D_range, D_range[-1]+k0))
    opts = Opts_COPAR(max_iter = 300, \
                lambda1     = 0.01,\
                eta         = 0.0001, \
                D_range_ext = D_range_ext,\
                verbose      = True,\
                check_grad  = False)
    Y_range = N*np.arange(C+1)
    Y = normc(np.random.rand(d, C*N))
    D = normc(np.random.rand(d, k*C + k0))
    X = np.random.rand(D.shape[1], Y.shape[1])
    # COPAR(Y, Y_range, opts)
    COPAR_updateX(Y, Y_range, D, X, opts)



# COPAR_test()


def COPAR_pred_GC(Y, D, gamma, opts):
    D_range_ext = opts.D_range_ext
    C = numel(D_range_ext) - 2
    optsX = Opts_COPAR(verbose = False, max_iter = 300)

    X = lasso_fista(Y, D, zeros(D.shape[1], Y.shape[1]), gamma, optsX)[0]
    DCp1_range = range(D_range_ext[-2], D_range_ext[-1])
    DCp1 = D[:, DCp1_range]
    XCp1  = X[DCp1_range, :]
    Ybar = Y - np.dot(DCp1, XCp1)
    E = zeros(C, Y.shape[1])
    for c in range(C):
        Xc = get_block_row(X, c, D_range_ext)
        Dc = get_block_col(D, c, D_range_ext)
        R = Ybar - np.dot(Dc, Xc)
        E[c, :] = (R*R).sum(axis = 0)

    pred = np.argmin(E, axis = 0) + 1
    return pred 

def COPAR_pred_LC(Y, D, gamma, opts):
    D_range_ext = opts.D_range_ext
    optsX = Opts_COPAR(verbose = False, max_iter = 300)
    DCp1_range = range(D_range_ext[-2], D_range_ext[-1])
    DCp1 = D[:, DCp1_range]
    C = numel(D_range_ext) - 2
    E = zeros(C, Y.shape[1])
    for c in range(C):
        # Dc_range = range(D_range_ext[c]
        Dchat = get_block_col(D, [c, C], D_range_ext)
        X = lasso_fista(Y, Dchat, np.array([]), gamma, optsX)[0]
        R1 = Y - np.dot(Dchat, X)
        R2 = gamma*abs(X)
        E[c, :] = (R1*R1).sum(axis = 0) + (R2*R2).sum(axis = 0)

    pred = np.argmin(E, axis = 0) + 1
    return pred 




def COPAR_top(dataset, n_c, k, k0, alambda, eta, verbose = False, show_progress = True):
    """
    COPAR_top(dataset, n_c, k, alambda, eta)
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/19/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """
    print "===============COPAR==============="
    print "Apply COPAR on " + dataset + " with parameters:"
    print "n_c: ", n_c, '\nk: ', k, '\nlambda: ',\
            alambda, '\neta: ', eta
    print '-------------------'
    ## get data
    # print "\nPreparing training and test samples...",
    dataset, Y_train, Y_test, label_train, label_test = \
        train_test_split(dataset, n_c)
    ## Prepare parameters
    C = np.unique(label_train).size 
    D_range    = k*np.arange(C+1)
    D_range_ext = np.hstack((D_range, D_range[-1]+k0))
    opts = Opts_COPAR(  max_iter = 20, \
                        lambda1     = alambda, \
                        D_range_ext = D_range_ext,\
                        verbose     = verbose,\
                        eta         = eta)
    Y_range     = label_to_range(label_train)
    ## Train 
    print "Training ..."
    D, X = COPAR(Y_train, Y_range, opts, show_progress)
    print "...done training"
    ##
    print "Test...",
    print "GC: "
    # for gamma in [0.001, 0.005, 0.01]:
    for gamma in [0.001]:
        print 'gamma = %.4f' % gamma,
        pred = COPAR_pred_GC(Y_test, D, gamma, opts)
        acc = calc_acc(pred, label_test)
        print '| acc = %2.2f%%' % (100 * acc)

    # print "LC: "
    # for gamma in [0.001, 0.005, 0.01]:
    #     print 'gamma = %.4f' % gamma,
    #     pred = COPAR_pred_LC(Y_test, D, gamma, opts)
    #     acc = calc_acc(pred, label_test)
    #     print '| acc = %2.2f%%' % (100 * acc)

    return np.amax(acc)
    print "...done test"

def COPAR_top_test():
    COPAR_top('myYaleB', 3, 2, 2, 0.001, 0.01)
    # COPAR_top('myARgender', 10, 8, 5, 0.001, 0.01)

# COPAR_top_test()

def COPAR_updateD_test():
    fn = os.path.join('data', 'tmp2.pickle')
    Vars = myload(fn)

    # print Vars.keys()

    D = Vars['D']
    Y = Vars['Y']
    X = Vars['X']
    Y_range = Vars['Y_range'][0]

    print Y_range[0]


    C = 10
    N = 7
    d = 300 
    k = 7 
    k0 = 10
    D_range = np.arange(C+1)*k 
    D_range_ext = np.hstack((D_range, D_range[-1] + k0))
    opts = Opts_COPAR(max_iter = 100, verbose = True, lambda1 = 0.01, eta = 0.1, \
        D_range_ext = D_range_ext)
    COPAR_updateD(Y, Y_range, D, X, opts) 

# COPAR_updateD_test()
def COPAR_test():
    fn = os.path.join('data', 'tmp3.pickle')
    Vars = myload(fn)

    # print Vars.keys()

    Y = Vars['Y']
    Y_range = Vars['Y_range'][0]
    C = 10 
    N = 7 
    d = 30 
    k = 7
    k0 = 10 
    D_range     = k*np.arange(C+1)
    D_range_ext = np.hstack((D_range, D_range[-1]+k0))

    opts = Opts_COPAR(lambda1 = 0.0001, eta = 0.01, max_iter = 10, \
        verbose = True, D_range_ext = D_range_ext)

    COPAR(Y, Y_range, opts)

# COPAR_test()
def COPAR_updateX_test():
    fn = os.path.join('data', 'tmp5.pickle')
    Vars = myload(fn)

    # print Vars.keys()

    Y = Vars['Y']
    Y_range = Vars['Y_range'][0]
    X = Vars['X']
    D = Vars['D']

    d  = 30
    N  = 7
    k  = 7
    k0 = 10
    C  = 20

    D_range     = k*np.arange(C+1)
    D_range_ext = np.hstack((D_range, D_range[-1]+k0))

    opts = Opts_COPAR(lambda1 = 0.0001, eta = 0.01, max_iter = 10, \
        verbose = True, D_range_ext = D_range_ext)

    COPAR_updateX(Y, Y_range, D, X, opts)

# COPAR_updateX_test()

def COPAR_updateXc_test():
    fn = os.path.join('data', 'tmp.pickle')
    Vars = myload(fn)

    # print Vars.keys()

    D = Vars['D']
    Y = Vars['Y']
    Xc = Vars['Xc']
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
    D_range = D_range_ext[: -1]

    opts = Opts_COPAR(max_iter = 300, verbose = True, tol = 1e-8, check_grad = False,\
        D_range_ext = D_range_ext, lambda1 = 0.01, eta = 0.1)

    DCp1 = get_block_col(D, C, D_range_ext)
    DCp1tDCp1 = np.dot(DCp1.T, DCp1)

    k0 = D_range_ext[-1] - D_range_ext[-2]
    if k0 > 0: 
        L = max_eig(DtD) + max_eig(DCp1tDCp1)
    else:
        L = max_eig(DtD)

    Xc = COPAR_updateXc(DtD, DCp1tDCp1, DtY,  Y_range, Xc, c, L, opts)

# COPAR_top('myYaleB', 10, 8, 5, 0.001, 0.01)