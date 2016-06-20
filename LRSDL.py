from utils import * 
from ODL import *
from FDDL import *
from my_opti import *
from matlab_syntax import *
class Opts_LRSDL:
    def __init__(self, verbal = False, max_iter = 100, tol = 1e-8, k = 0, \
        k0 = 0, D_range_ext = np.array([0]), lambda1, lambda2,\
        eta, check_grad = False):
        self.verbal      = verbal
        self.max_iter    = max_iter
        self.tol         = tol
        self.k           = k
        self.k0          = k0
        self.D_range_ext = D_range_ext
        self.lambda1     = lambda1
        self.lambda2     = lambda2
        self.eta         = eta
        self.check_grad  = check_grad
#################### LRSDL #########################
def LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts):
    """
    Syntax: cost = LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts)
    Calculate cost of LRSDL, include 5 terms:
        + ||Y - D0X0 - DX||_F^2 
        + FDDL_fidelity with Ybar = Y - D0X0 
        + FDDL_discriminative
        + ||X0 - M0||_F^2 
        + ||D0||_*
        + ||X||_1 + ||X0||-1
    """ 
    Ybar = Y - np.dot(D0, X0) 
    cost = 0.5*normF2(Ybar - np.dot(D, X)) + \
           0.5*FDDL_fidelity(Ybar, Y_range, D, D_range, X) + \
           opts.lambda1*norm1(X) + opts.lambda1*norm1(X0) + \
           0.5*opts.lambda2*(FDDL_discriminative(X, Y_range) + \
                normF2(X0 - build_mean_matrix(X0))) + \
           opts.eta*nuclearnorm(D0)
    return cost 

def LRSDL_updateXX0(Y, Y_range, D, D_range, D0, X, X0, opts):
    K = D_range[-1]
    
    def extractFromX1(X1):
        X = X1[:K, :]
        X0 = X1[K:, :]
        return (X, X0)
    # cost 
    def calc_f(X1):
        X, X0 = extractFromX1(X1)
        Ybar = Y - np.dot(D0, X0)
        cost = 0.5*(normF2(Ybar - np.dot(D, X)) + \
                    FDDL_fidelity(Ybar, Y_range, D, D_range, X)) + \
                0.5*opts.lambda2* (FDDL_discriminative(X, Y_range) + \
                    normF2(X0 - buildMean(X0)))
        return cost 
    # total cost 
    def calc_F(X1):
        return calc_f(X1) + opts.lambda1*norm1(X1)
    # gradient for fista 
    def grad(X1):
        X, X0 = extractFromX1(X1)
        DtY   = DtY0 - np.dot(DtD0, X0)
        Y_0   = buildMhat(DtY, D_range, Y_range)
        g     = np.dot(Dhat, X) - Y_0 + buildM_2Mbar(X, Y_range, lambda2);
        g0    = np.dot(A, X0) - D0tY2 + np.dot(D0tD, buildMhat(X, D_range, Y_range)) \
                - opts.lambda2*buildMean(X0)
        return  np.vstack((g, g0))
        # return g1 
    #
    if D0.shape[1] == 0:
        X = FDDL_updateX(Y, Y_range, D, D_range, X, opts)
        X0 = np.array([])
    else:
        lambda1 = opts.lambda1
        lambda2 = opts.lambda2
        DtD     = np.dot(D.T, D)
        D_0     = buildMhat(DtD, D_range, D_range)
        Dhat    = D_0 + 2*opts.lambda2*eye(D_0.shape[0])
        D0tD0   = np.dot(D0.T, D0)
        A       = 2*D0tD0 + opts.lambda2*eye(D0.shape[1])
        DtY0    = np.dot(D.T, Y)
        DtD0    = np.dot(D.T, D0)
        D0tD    = DtD0.T
        D0tY2   = 2*np.dot(D0.T, Y)
        # check gradient
        if opts.check_grad and not check_grad(calc_f, grad, np.vstack((X, X0))):
            print 'Check gradient or cost again!'
        # main fista 
        optsXX0 = Opts(max_iter = 200)
        L       = max_eig(Dhat) + max_eig(A) + 4*opts.lambda2 + 1;
        X1      = np.vstack((X, X0))
        X1      = fista(grad, X1, L, opts.lambda1, optsXX0, calc_F)[0]
        X, X0   = extractFromX1(X1)
    return (X, X0)


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
    D0 = min_rank_dict0(L, X0, opts.eta/2, D0, opts)
    return D0 

    # pass 

def LRSDL_init(Y, Y_range, opts):
    """
    D, D0, X, X0 = LRSDL_init(Y, Y_range, D_range_ext, opts_init)
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/20/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """ 
    D_range_ext = opts.D_range_ext
    C = Y_range.size - 1
    print C
    D = np.zeros((Y.shape[0], D_range_ext[-2]))
    X = np.zeros((D_range_ext[-2], Y.shape[1]))
    ## class-specific dictionary 
    optsODL = Opts(max_iter = 50, tol = 1e-8, show_cost = 0)
    for c in xrange(C):
        if opts.verbal:
            print '%3d' %(c+1), 
            if c%10 == 9:
                print '\n',
        Yc = get_block_col(Y, c, Y_range)
        D[:, D_range_ext[c]:D_range_ext[c+1]], Xcc = \
            ODL(Yc, D_range_ext[c+1] - D_range_ext[c], opts.lambda1, optsODL)
        X[D_range_ext[c]:D_range_ext[c+1], Y_range[c]:Y_range[c+1]] = \
            Xcc.copy()
    if opts.verbal:
        print '\n',
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
                eta = 0.0001, \
                D_range_ext = D_range_ext,
                check_grad = False)
    LRSDL_init(Y, Y_range, opts)

def LRSDL(Y, Y_range, opts):
    """
    D, D_range, D0, X, X0, M, m0 = LRSDL(Y, Y_range, D_range_ext, opts)
    """ 
    print 'hehe'
    D_range_ext = opts.D_range_ext
    D_range = D_range_ext[:-1].copy()
    k0 = D_range_ext[-1] - D_range_ext[-2]
    ## Initialization 
    
    # opts_init = opts.copy()
    opts_init = Opts_LRSDL( max_iter    = 30, \
                            lambda1     = opts.lambda1,\
                            lambda2     = opts.lambda2,\
                            eta         = opts.eta,\
                            D_range_ext = opts.D_range_ext)

    D, D0, X, X0 = LRSDL_init(Y, Y_range, opts_init)
    cost_init =  LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts)
    print 'Initial cost: ', cost_init
    ## 
    # optsXX0 = opts.copy()
    # optsXX0.max_iter = 300 
    optsXX0 = Opts_LRSDL(max_iter = 300,\
                        lambda1 = opts.lambda1,\
                        lambda2 = opts.lambda2)
    # optsD = opts.copy()
    # optsD.max_iter = 200 
    optsD = Opts_LRSDL(max_iter = 300, lambda1 = opts.lambda1, lambda2 = opts.lambda2, \
            eta = opts.eta, D_range_ext = opts.D_range_ext)
    # optsD0 = opts.copy()
    # opts.max_iter 
    # optsD0.max_iter = 100
    optsD0 = Opts_LRSDL(max_iter = 100, lambda1 = opts.lambda1, lambda2 = opts.lambda2, \
            eta = opts.eta, D_range_ext = opts.D_range_ext)

    tol_XX0 = 1e-5 
    it = 0

    is_fddl = k0 == 0
    print opts.max_iter 
    while it < opts.max_iter:
        it += 1 
        if opts.verbal:
            print 'iter %3d/%3d |'% (it, opts.max_iter)
            print 'updating XX0...',
        ## updateXX0 
        if is_fddl:
            X = FDDL_updateX(Y, Y_range, D, D_range, X, optsXX0)
        else:
            # X1 = LRSDL_updateXX0(Y, Y_range, D, D_range, D0, X, X0, optsXX0)

            X, X0 = LRSDL_updateXX0(Y, Y_range, D, D_range, D0, X, X0, optsXX0)
            # X = X1[0]
            # X0 = X1[1]

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
        if opts.verbal:
            print 'costX ',  LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts)
            ## update D 
            print 'updating D  ...',
        D, _ = LRSDL_updateD_fast(Y, Y_range, D, D_range, D0, X, X0, optsD)
        if opts.verbal: 
            print 'costD ',  LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts)
        ## update D0
        if not is_fddl:
            if opts.verbal: 
                print 'updating D0 ...',
            D0 = LRSDL_updateD0(Y, Y_range, D, D_range, D0, X, X0, optsD0)
            # print D0.shape
            if opts.verbal: 
                print 'costD0', LRSDL_cost(Y, Y_range, D, D_range, D0, X, X0, opts)

    M = build_mean_vector(X, Y_range)

    if is_fddl:
        m0 = None 
    else:
        m0 = np.mean(X0, axis = 1)
    return (D, D0, X, X0, M, m0, opts)

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
                eta = 0.0001, \
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
def LRSDL_pred_GC(Y, D, D0, CoefM, m0, opts, label_test):
    # pass
    print "LRSDL_pred_GC, size D = ", D.shape[1]
    C = CoefM.shape[1]
    k = D.shape[1]
    k0 = opts.k0 
    # D_range = k*(0:n)
    D_range = opts.D_range_ext[:-1]

    N = Y.shape[1]
    acc = np.array([])

    def local_sparse_coding(Y, D, D0, m0, lambda1, lambda2):
        N      = Y.shape[1]
        k      = D.shape[1]
        k0     = D0.shape[1]
        X1init = np.zeros([k + k0, N])
        D1     = np.hstack((D, D0))
        M0     = repmat(m0, 1, N)
        D1tD1  = np.dot(D1.T, D1)
        D1tY   = np.dot(D1.T, Y)
        # cost 
        def calc_F(X1):
            X    = X1[:k, :]
            X0   = X1[k:, :]
            cost = 0.5*normF2(Y - np.dot(D1, X1)) + \
                    0.5*lambda2*normF2(X0 - M0) + \
                    lambda1*norm1(X1);
            return cost 
        # grad
        def grad(X1):
            X = X1[:k, :]
            X0 = X1[k:, :]
            g = np.dot(D1tD1, X1) - D1tY + lambda2*\
                np.vstack((zeros(k, N), X0 - M0))
            return g 
        # L = np.max(LA.eig(D1tD1))[0][0]
        L    = max_eig(D1tD1)
        opts = Opts(max_iter                                    = 300)
        X1   = fista(grad, X1init, L, lambda1, opts, calc_F)[0]
        X    = X1[:k, :]
        X0   = X1[k:,:]
        return (X, X0)

    for lambda1 in [0.0001, 0.01]:
        X, X0 = local_sparse_coding(Y, D, D0, m0, lambda1, 0.01)
        Yhat  = Y - np.dot(D0, X0)
        E1    = zeros(C, N)
        E2    = zeros(C, N)
        for c in xrange(C):
            Dc = get_block_col(D, c, D_range)
            Xc = get_block_row(X, c, D_range)
            Mc = repmat(CoefM[:, c], 1, N)
            R1 = Yhat - np.dot(Dc, Xc)
            R2 = X - Mc 
            E1[c,:] = (R1*R1).sum(axis = 0)
            E2[c,:] = (R2*R2).sum(axis = 0)
        for w in [.25, .5, .75]:
            E    = w*E1 + (1 - w)*E2
            pred = np.argmin(E, axis          = 0) + 1
            aaaa = calc_acc(pred, label_test)
            acc  = np.hstack((acc, aaaa))
            print 'w = %2f' %w, '| l1 = %.4f' %lambda1,'| acc = %.2f' %(100*aaaa) 







def LRSDL_top(dataset, n_c, k, k0, lambda1, lambda2, eta):
    """
    * Syntax `LRSDL_top(dataset, n_c, k, k0, lambda1, lambda2, eta)`
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/19/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """
    print "Apply LRSDL on " + dataset + "with parameters:"
    print "n_c: ", n_c, '\nk: ', k, '\nk0: ', k0, '\nlambda1: ',\
            lambda1, '\nlambda2: ', lambda2, '\neta: ', eta
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
        '_l2_' + str(lambda2) + '_l3_' + str(eta) + '_' +t + '.pickle')
    output_file = open(fn, 'w+')
    print 'LRSDL on', dataset 
    ## Prepare parameters
    C = np.unique(label_train).size 
    D_range     = k*np.arange(C+1)
    D_range_ext = np.hstack((D_range, D_range[-1]+k0))
    Y_range     = label_to_range(label_train)

    opts = Opts_LRSDL(max_iter  = 20, \
                    D_range_ext = D_range_ext,\
                    lambda1     = lambda1, \
                    lambda2     = lambda2, \
                    verbal      = True,
                    eta         = eta)
    ## Train 
    print "Training ...",
    # D, D_range, D0, X, X0, M, m0 = LRSDL(Y_train, Y_range, D_range_ext, opts)
    D, D0, X, X0, M, m0, opts = LRSDL(Y_train, Y_range, opts)
    print "...done training"
    ##
    X1 = np.vstack((X, X0))
    CoefMM0 = zeros(X1.shape[0], C)
    for c in xrange(C):
        X1c = get_block_col(X1, c, Y_range) 
        CoefMM0[:, c] = np.mean(X1c, axis = 1)
    ## Test 
    print "Test...",
    # acc = LRSDL_pred(Y_test, D,  D_range, D0, CoefMM0, label_test)
    # acc = 
    print D.shape
    LRSDL_pred_GC(Y_test, D, D0, M, m0, opts, label_test)
    print "...done test"
    ## save results
    # A = {'acc': acc}
    # cPickle.dump(A, output_file)
    # close(output_file)

def LRSDL_top_test():
    # LRSDL_top(dataset, N_train, k, k0, lambda1, lambda2, eta)
    
    LRSDL_top('myYaleB', 2, 1, 1, 0.001, 0.01, .1)
    # LRSDL_top('myYaleB', 15, 10, 5, 0.001, 0.01, .1)
# LRSDL_init_test()

