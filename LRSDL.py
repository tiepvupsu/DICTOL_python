from utils import * 
from ODL import *

def LRSDL_updateXX0(Y, Y_range, D, D_range, D0, X, X0, opts):
    """
    (X, X0, it) = LRSDL_updateXX0(Y, Y_range, D, D_range, D0, X, X0, opts)
    update X and X0 in LRSDL using FISTA algorithm.
    
    Note: X1 = np.vstack((X, X0))
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/21/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """
    X1    = np.vstack((X, X0))
    C     = Y_range.size - 1
    DtD   = np.dot(D.T, D) 
    D_0   = buildMhat(DtD, D_range, D_range)
    Dhat  = D_0 + 2*opts.lambda2*np.eye(D_0.shape[0])
    DtD0  = np.dot(D.T, D0)
    D0tD  = DtD0.T.copy()
    D0tD0 = np.dot(D0.T, D0)
    DtY0  = np.dot(D.T, Y)
    D0tY2 = 2*np.dot(D0.T, Y)

    A = 2*D0tD0 + opts.lambda2*np.eye(D0.shape[1])

    ## DtYmask 
    DtYmask = np.zeros((D.shape[1], Y.shape[1]))
    DtDmask = np.zeros((D.shape[1], D.shape[1]))
    for c in xrange(C):
        k = D_range[c+1] - D_range[c] # number of class-specific bases 
        N = Y_range[c+1] - Y_range[c]
        DtYmask[D_range[c]: D_range[c+1], Y_range[c]: Y_range[c+1]] = \
            np.ones((k, N))
        DtDmask[D_range[c]: D_range[c+1], D_range[c]: D_range[c+1]] = \
            np.ones((k, k))
    # print DtYmask 
    # time.sleep(5)
    # X, X0 = extractFromX1(X1):
    def extractFromX1(X1):
        X  = X1[:D_range[-1], :]
        X0 = X1[D_range[-1]:, :]        
        return (X, X0)

    ## calculate cost
    def calc_f(X1):
        X, X0 = extractFromX1(X1)
        cost  = LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts) -\
                opts.lambda1*norm1(X1)
        return cost 

    def calc_F(X1):
        X, X0 = extractFromX1(X1) 
        cost  = LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts)
        return cost 

    ## gradient     
    def grad(X1):
        X, X0 = extractFromX1(X1)
        DtY   = DtY0 - np.dot(DtD0, X0)
        Y_0   = buildMhat(DtY, D_range, Y_range)
        g     = np.dot(Dhat, X) - Y_0 + buildM_2Mbar(X, Y_range, opts.lambda2)
        g0    = np.dot(A, X0) - D0tY2 + np.dot(D0tD, X + X*DtYmask) - \
                opts.lambda2*buildMean(X0) 
        g1    = np.vstack((g, g0))
        return g1 

    if opts.check_grad:
        if not check_grad(calc_f, grad, X1):
            exit()

    optsXX0 = opts 
    optsXX0.max_iter = 300 
    ## fista
    L = max(LA.eig(Dhat)[0]) + max(LA.eig(A)[0]) + 6*opts.lambda2 + 1 
    X1, it  = fista(grad, X1, L, opts.lambda1, optsXX0, calc_F)
    ###
    X, X0 = extractFromX1(X1)
    return (X, X0, it)
    # pass

def LRSDL_updateD(Y, Y_range, D, D_range, D0, X, X0, opts):
    pass 

def LRSDL_updateD_fast(Y, Y_range, D, D_range, D0, X, X0, opts):
    if D0.shape[1] == 0:
        Ybar = Y
    else:
        Ybar = Y - np.dot(D0, X0)
    return FDDL_updateD_fast(Y, Y_range, D, D_range, X, opts)
    # pass 

def LRSDL_buildYhat(Y, Y_range, D, D_range, X):
    """
    Yhat = LRSD_buildYhat(Y, Y_range, D, D_range, X)
    Yhat = [Yhat_1, Yhat_2, ..., Yhat_C]
    where Yhat_c = Yc - Dc*Xcc 
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/22/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """
    C = Y_range.size - 1 
    Yhat = np.zeros_like(Y)
    for c in xrange(C):
        Yc = get_block_col(Y, c, Y_range)
        Dc = get_block_col(D, c, D_range)
        Xcc = get_block(X, c, c, D_range, Y_range)
        Yhat[:, Y_range[c]: Y_range[c+1]] = Yc - np.dot(Dc, Xcc)

    return Yhat 

def LRSDL_updateD0(Y, Y_range, D, D_range, D0, X, X0, opts):
    """
    D0 = LRSDL_updateD0(Y, Y_range, D, D_range, D0, X, X0, opts):
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/22/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """ 
    Ybar = Y - np.dot(D, X) 
    Yhat = LRSDL_buildYhat(Y, Y_range, D, D_range, X)

    L = (Ybar + Yhat)/2 
    D0 = min_rank_dict0(L, X0, opts.lambda3/2, D0, opts)
    return D0 

    # pass 

def LRSDL_init(Y, Y_range, D_range_ext, opts):
    """
    D, D0, X, X0 = LRSDL_init(Y, Y_range, D_range_ext, opts_init)
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/20/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """ 
    print Y.shape
    print 'C = ',

    C = Y_range.size - 1
    print C
    D = np.zeros((Y.shape[0], D_range_ext[-2]))
    X = np.zeros((D_range_ext[-2], Y.shape[1]))
    ## class-specific dictionary 
    optsODL = Opts(max_iter = 50, tol = 1e-8, show_cost = 0)
    for c in xrange(C):
        print c, 
        Yc = get_block_col(Y, c, Y_range)
        D[:, D_range_ext[c]:D_range_ext[c+1]], Xcc = \
            ODL(Yc, D_range_ext[c+1] - D_range_ext[c], opts.lambda1, optsODL)
        X[D_range_ext[c]:D_range_ext[c+1], Y_range[c]:Y_range[c+1]] = \
            Xcc.copy()
    ## shared dictionary 
    k0 = D_range_ext[-1] - D_range_ext[-2]
    if k0 > 0:
        D0, X0 = ODL(Y, k0, opts.lambda1, opts)
    return (D, D0, X, X0)

def LRSDL_init_test():
    Y_range = np.array([0, 10, 20])
    D_range_ext = np.array([0, 2, 4, 6])
    Y = np.random.rand(10, Y_range[-1])
    opts = Opts(max_iter = 2, \
                lambda1 = 0.01,\
                lambda2 = 0.01, \
                lambda3 = 0.0001, \
                check_grad = False)
    LRSDL_init(Y, Y_range, D_range_ext, opts)

def LRSDL(Y, Y_range, D_range_ext, opts):
    """
    D, D_range, D0, X, X0, M, m0 = LRSDL(Y, Y_range, D_range_ext, opts)
    """ 
    print 'hehe'
    D_range = D_range_ext[:-1].copy()
    k0 = D_range_ext[-1] - D_range_ext[-2]
    ## Initialization 
    opts_init = opts.copy()
    opts_init.max_iter = 30 
    D, D0, X, X0 = LRSDL_init(Y, Y_range, D_range_ext, opts_init)
    cost_init =  LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts)
    print 'Initial cost: ', cost_init
    ## 
    optsXX0 = opts.copy()
    optsXX0.max_iter = 300 
    optsD = opts.copy()
    optsD.max_iter = 200 
    optsD0 = opts.copy()
    opts.max_iter 
    optsD0.max_iter = 100

    tol_XX0 = 1e-5 
    it = 0

    is_fddl = k0 == 0
    print opts.max_iter 
    while it < opts.max_iter:
        it += 1 
        print 'iter %3d/%3d |'% (it, opts.max_iter)
        ## updateXX0 
        print 'updating XX0...',
        if is_fddl:
            X = FDDL_updateX(Y, Y_range, D, D_range, X, optsXX0)
        else:
            X1 = LRSDL_updateXX0(Y, Y_range, D, D_range, D0, X, X0, optsXX0)
            X = X1[0]
            X0 = X1[1]
            # os.system("pause")
            # time.sleep(5)
            ## reduce shared dictionary
            g0 = np.sum(abs(X0), axis = 1)
            unused_id0 = np.nonzero(g0 < tol_XX0)
            D0 = np.delete(D0, unused_id0, axis = 1)
            X0 = np.delete(X0, unused_id0, axis = 0)
            if D0.shape[1] == 0:
                is_fddl = True 
        ## reduce normal dictionaries 
        g = np.sum(abs(X), axis = 1) 
        unused_id = np.nonzero(g < tol_XX0)
        D = np.delete(D, unused_id, axis = 1)
        X = np.delete(X, unused_id, axis = 0) 
        range_reduce(D_range, unused_id)
        print 'costX ',  LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts)
        ## update D 
        print 'updating D  ...',
        D, _ = LRSDL_updateD_fast(Y, Y_range, D, D_range, D0, X, X0, optsD)
        print 'costD ',  LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts)
        ## update D0
        if not is_fddl:
            print 'updating D0 ...',
            D0 = LRSDL_updateD0(Y, Y_range, D, D_range, D0, X, X0, optsD0)
            # print D0.shape
            print 'costD0', LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts)

    M = build_mean_vector(X, Y_range)
    if is_fddl:
        m0 = None 
    else:
        m0 = np.mean(X0, axis = 0)
    return (D, D_range, D0, X, X0, M, m0)

def LRSDL_test():
    print '-----------------------------------------'
    print '`LRSDL` test: '
    C = 3
    d = 30 
    n = 20 
    k =  5
    k0 = 5 
    opts = Opts(max_iter = 2, \
                lambda1 = 0.01,\
                lambda2 = 0.01, \
                lambda3 = 0.0001, \
                check_grad = False)
    D_range = k*np.arange(C+1)
    D_range_ext = np.hstack((D_range, D_range[-1]+k0))
    # print D_range, D_range_ext 
    Y_range = n*np.arange(C+1)
    Y = normc(np.random.rand(d, C*n))
    LRSDL(Y, Y_range, D_range_ext, opts)

def LRSDL_pred(Y, D,  D_range, D0, CoefMM0, label_test):
    """
    acc = LRSDL_pred(Y_test, D,  D_range, D0, CoefMM0, label_test):
    """
    C = CoefMM0.shape[1]
    N = Y.shape[1]
    acc = np.array([])
    optsX = Opts(max_iter = 500) 
    lambda_list = [0.0001, .001, 0.005, 0.01]
    for alambda in lambda_list:
        E = zeros(C, Y.shape[1])
        for c in xrange(C):
            Dc = np.hstack((get_block_col(D, c, D_range), D0))
            Xc, _ = lasso_fista(Y, Dc, np.array([]), alambda, optsX)
            R = Y - np.dot(Dc, Xc) 
            E[c, :] = 0.5*np.sum(R*R, axis = 0) + \
                      alambda*np.sum(np.abs(Xc), axis = 0)
        pred = np.argmin(E, axis = 0) + 1 
        acc0 = float(np.nonzero(pred == label_test)[0].size)/label_test.size 
        print '\nlambda = %f' %alambda, '| acc = ', acc0
        acc = np.hstack((acc, acc0))
    return acc 

    # return pred 
            # pass 

def LRSDL_top(dataset, n_c, k, k0, lambda1, lambda2, lambda3):
    """
    * Syntax `LRSDL_top(dataset, n_c, k, k0, lambda1, lambda2, lambda3)`
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/19/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """
    print "Apply LRSDL on " + dataset + "with parameters:"
    print "n_c: ", n_c, '\nk: ', k, '\nk0: ', k0, '\nlambda1: ',\
            lambda1, '\nlambda2: ', lambda2, '\nlambda3: ', lambda3
    ## get data
    print "\nPreparing training and test samples...",
    dataset, Y_train, Y_test, label_train, label_test = \
        train_test_split(dataset, n_c)
    print "done"
    ## output filename 
    path = 'results/LRSDL'
    if not os.path.exists(path):
        os.makedirs(path)
    t = get_time_str()
    fn = os.path.join('results', 'LRSDL', dataset + '_N_'+ str(n_c) + \
        '_k_' + str(k) + '_k0_' + str(k0) + '_l1_' + str(lambda1) + \
        '_l2_' + str(lambda2) + '_l3_' + str(lambda3) + '_' +t + '.pickle')
    output_file = open(fn, 'w+')
    print 'LRSDL on', dataset 
    ## Prepare parameters
    C = np.unique(label_train).size 
    opts = Opts(max_iter = 100, \
                lambda1  = lambda1, \
                lambda2  = lambda2, \
                lambda3  = lambda3)
    D_range     = k*np.arange(C+1)
    D_range_ext = np.hstack((D_range, D_range[-1]+k0))
    Y_range     = label_to_range(label_train)
    ## Train 
    print "Training ...",
    D, D_range, D0, X, X0, M, m0 = LRSDL(Y_train, Y_range, D_range_ext, opts)
    print "...done training"
    ##
    X1 = np.vstack((X, X0))
    CoefMM0 = zeros(X1.shape[0], C)
    for c in xrange(C):
        X1c = get_block_col(X1, c, Y_range) 
        CoefMM0[:, c] = np.mean(X1c, axis = 1)
    ## Test 
    print "Test...",
    acc = LRSDL_pred(Y_test, D,  D_range, D0, CoefMM0, label_test)
    print "...done test"
    ## save results
    A = {'acc': acc}
    cPickle.dump(A, output_file)
    close(output_file)

def LRSDL_top_test():
    LRSDL_top(dataset, N_train, k, k0, lambda1, lambda2, lambda3)
    
LRSDL_top('myYaleB', 5, 2, 2, 0.001, 0.01, .1)
# LRSDL_init_test()