# import numpy as np 
# from numpy import linalg as LA
from matlab_syntax import *
# import scipy.io as sio
import cPickle
import math 
import time 
from time import strftime
# from six.moves import cPickle as pickle
import os 
import io 
import scipy.io as sio
import pickle
from ODL import *
test = True 
# test = False 


def get_time_str():
    print 'Time now: ' + strftime("%m/%d/%Y %H:%M:%S")
    return strftime("%m%d_%H%M%S")

def myreshape(x, c, r):
    return x.reshape(c, r, order = 'F').copy()

def label_to_range(label):
    """
    * Convert from Labels to Ranges
    * Example: if `label = [1 1 1 2 2 2 2 3 3]`, then `range = [0, 3, 7, 9]`.
    * Syntax: `arange = label_to_range(label)`
        - `label`: a numpy array 
        - `arange`: a numpy array
    """
    C = int(label.max())
    arange = np.zeros((C+1,), dtype=np.int)
    cumsum = 0
    for i in xrange(C):
        cumsum += np.where(label == (i+1))[0].size 
        arange[i+1] = cumsum 
    return arange

def label_to_range_test():
    label = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3])
    print "---------------------------------------\nLabel_to_range test:"
    print "label = ", label 
    print "range = ", label_to_range(label)
    # ouput should be [0, 3, 7, 9]
    # pass 

def range_to_label(arange):
    """
    * Convert from Ranges to Labels
    * Example: if `range = [0, 3, 5]`` then `label = [1 1 1 2 2]``
    * Syntax: `label = range_to_label(range)`
    """
    # pass 
    C = arange.size - 1 
    label = np.ones((arange[-1], ), dtype=np.int)
    for i in xrange(1, C):
        n = arange[i] - arange[i-1]
        label[arange[i]: arange[i+1]] *= (i+1) 
    return label

def range_to_label_test():
    arange = np.array([0, 3, 5, 9])
    print "---------------------------------------\n`range_to_label` test:"
    print "range: ", arange 
    print "label: ", range_to_label(arange) 

def get_block_col(M, c, col_range):
    """
    * Syntax: `Mc = get_block_col(M, c, col_range)`
    * Extract a block of columns from a matrix.
        - `M`: the big matrix `M = [M_1, M_2, ...., M_C]`.
        - `c`: block index (start at 0).
        - `col_range`: range of samples, see `Y_range` and `D_range` above.
    * Example: `M` has 25 columns and `col_range = [0, 10, 25]`, then 
    `get_block_col(M, 1, col_range)` will output the first block of `M`, 
    i.e. `M(:, 1:10)`.
    """
    return M[:, col_range[c]: col_range[c+1]]
    # pass 

def get_block_col_test():
    print "---------------------------------------\n`get_block_col test:"
    A = np.random.randint(5, size=(3, 9))
    arange = np.array([0, 4, 7, 9], dtype = np.int)
    print " A: \n", A
    print " arange: ", arange  
    print " get_block_col(A, 2, arange): \n", get_block_col(A, 2, arange)

def get_block_row(M, c, row_range):
    """
    * Extract a block of rows from a matrix.
    * Syntax: `Mc = get_block_row(M, c, row_range)`
        - `M`: the big matrix `M = [M_1; M_2; ....; M_C]`.
        - `c`: block index (start at 0).
        - `row_range`: range of samples, see `Y_range` and `D_range` above.
    * Example: `M` has 40 rows and `row_range = [0, 10, 25, 40]`, then 
    `get_block_row(M, 2, row_range)` will output the second block of `M`, 
    i.e. `M(11:25, :)`.
    """
    return M[row_range[c]: row_range[c+1], :].copy()
    # pass 

def get_block_row_test():
    print "---------------------------------------\n`get_block_row` test:"
    A = np.random.randint(5, size=(9, 3))
    arange = np.array([0, 4, 7, 9], dtype = np.int)
    print " A: \n", A
    print " arange: ", arange  
    print " get_block_row(A, 3, arange): \n", get_block_row(A, 3, arange)

def get_block(M, i, j, row_range, col_range):
    """
    * Extract a submatrix of a matrix 
    * Syntax: `Mij = get_block(M, i, j, row_range, col_range)`
        - `M` the big matrix: 
        `M = [  M11, M12, ..., M1m; 
                M21, M22, ..., M2m; 
                ... ; 
                Mn1, Mn2, ..., Mnm]`
        - `i`: row block index 
        - `j`: column block index 
        - `row_range`: row range
        - `col_range`: columns range 
    * Note: `get_block(M, i, j, row_range, col_range) = 
    get_block_col(get_block_row(M, i, row_range), j, col_range).`
    """
    # pass 
    return M[row_range[i]:row_range[i+1], col_range[j]: col_range[j+1]].copy()

def get_block_test():
    print "---------------------------------------\n`get_block` test:"
    A = np.random.randint(5, size=(9, 9))
    row_range = np.array([0, 4, 7, 9], dtype = np.int)
    col_range = np.array([0, 2, 5, 9], dtype = np.int);
    print " A: \n", A
    print " row_range: ", row_range, "; col_range: ", col_range  
    print " get_block(A, 1, 2, row_range, col_range): \n", \
        get_block(A, 1, 2, row_range, col_range)



def norm1(X):
    """
    * Return norm 1 of a matrix, which is sum of absolute value of all element 
    of that matrix.
    """
    # pass 
    return abs(X).sum()
    # return LA.norm(X, 1)

def norm1_test():
    # pass 
    print('---------------------------------------')
    print '`norm1` test:'
    A = np.random.randint(8, size =(2,2)) - 3 
    print "A = \n", A 
    print "norm1(A) = ", norm1(A)

def normF2(X):
    """
    * Return square of the Frobenius norm, which is sum of square of all 
    elements in a matrix
    * Syntax: `res = normF2(X)`
    """
    # pass
    return LA.norm(X, 'fro')**2  

def normF2_test():
    print('---------------------------------------')
    print '`normF2` test:'
    A = np.random.randint(8, size =(2,2)) - 5
    print "A = \n", A 
    print "normF2(A) = ", normF2(A)

def is_vector(x):
    """
    check if a numpy.ndarray variable x is a vector
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/22/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """ 
    return len(x.shape) == 1 

def normc(A):
    # return A / LA.norm(A, axis=0)[None,:]
    return A/ np.tile(np.sqrt(np.sum(A*A, axis=0)), (A.shape[0], 1))

def nuclearnorm(X):
    """
    * Return nuclear norm of a matrix.
    * Syntax `res = nuclearnorm(X)`
    """

    return LA.norm(X) if is_vector(X) else LA.norm(X, 'nuc')


    pass 

def shrinkage(U, alambda):
    """
    * Soft thresholding function.
    * Syntax: ` X = shrinkage(U, lambda)`
    * Solve the following optimization problem:
    `X = arg min_X 0.5*||X - U||_F^2 + lambda||X||_1`
    where `U` and `X` are matrices with same sizes. `lambda` can be either 
    positive a scalar or a positive matrix (all elements are positive) with 
    same size as `X`. In the latter case, it is a weighted problem.
    """
    return np.maximum(0, U - alambda) + np.minimum(0, U + alambda)

def shrinkage_test():
    pass 

def shrinkage_rank(D, alambda):
    """
    * Singular value thresholding algorithm for matrix completion.
    * Syntax: `Y = shrinkage_rank(D, lambda)` 
    * Solve the following optimization problem:
      `X = arg min_X 0.5*||X - D||_F^2 + lambda*||X||_*`
      where `||X||_*` is the nuclear norm.
    """
    U, s, V = LA.svd(D, full_matrices=False)
    s1 = np.maximum(0, s - alambda)
    return np.dot(U, np.dot(np.diag(s1), V))


def shrinkage_rank_test():
    # pass 
    print('-------------------------------------------')
    print '`shrinkage_rank` test:'
    D = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    alambda = 0.1 
    print 'matlab result'
    print '    1.0586    1.9944    2.9302'
    print '    3.9944    4.9721    5.9498'
    print '    6.9302    7.9498    8.9693'
    print shrinkage_rank(D, alambda)


def min_rank_dict0_test():
    print '------------------------------------'
    print '`min_rank_dict0` test: '

    d = 10 
    N = 10 
    k = 5 
    Y = normc(np.random.rand(d, N))
    D = normc(np.random.rand(d, k))
    X = np.random.rand(k, N)
    opts = Opts()
    lambdaD = 0.01
    D = min_rank_dict0(Y, X, lambdaD, D, opts)

def fista(fn_grad, Xinit, L, alambda, opts, fn_calc_F):
    """
    * A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse 
            Problems.
    * Solve the problem: `X = arg min_X F(X) = f(X) + lambda||X||_1` where:
       - `X`: variable, can be a matrix.
       - `f(X)` is a smooth convex function with continuously differentiable 
       with Lipschitz continuous gradient `L(f)` 
       (Lipschitz constant of the gradient of `f`).
    * Syntax: `[X, iter] = fista(grad, Xinit, L, lambda, opts, calc_F)` where:
       - INPUT:
            + `grad`: a _function_ calculating gradient of `f(X)` given `X`.
            + `Xinit`: initial guess.
            + `L`: the Lipschitz constant of the gradient of `f(X)`.
            + `lambda`: a regularization parameter, can be either positive a 
                    scalar or a weighted matrix.
            + `opts`: a _structure_ variable describing the algorithm.
              * `opts.max_iter`: maximum iterations of the algorithm. 
                    Default `300`.
              * `opts.tol`: a tolerance, the algorithm will stop if difference 
                    between two successive `X` is smaller than this value. 
                    Default `1e-8`.
              * `opts.show_progress`: showing `F(X)` after each iteration or 
                    not. Default `false`. 
            + `calc_F`: optional, a _function_ calculating value of `F` at `X` 
                    via `feval(calc_F, X)`. 
      - OUTPUT:
        + `X`: solution.
        + `iter`: number of iterations.
    """ 
    Linv = 1/L
    lambdaLiv = alambda/L
    x_old = Xinit
    y_old = Xinit
    t_old = 1 
    it = 0
    cost_old = float("inf") # the positive infinity number 
    while it < opts.max_iter:
        it += 1 
        # t1 = time.time()
        x_new = np.real(shrinkage(y_old - Linv*fn_grad(y_old), lambdaLiv))
        # print x_new 
        # print type(x_new)
        # t2 = time.time() 
        # print t2 - t1
        t_new = 0.5*(1 + math.sqrt(1 + 4*t_old**2))
        y_new = x_new + (t_old - 1)/t_new * (x_new - x_old)
        e = norm1(x_new - x_old)/x_new.size 
        if e < opts.tol:
            break;
        x_old = x_new 
        t_old = t_new 
        y_old = y_new 
        if opts.verbal:
            cost_new = fn_calc_F(x_new)
            if cost_new <= cost_old:
                stt = 'YES.'
            else:
                stt = 'No, check your code.'
            print 'iter = '+str(it)+', cost = %4.4f' % cost_new, \
                ', cost decreases? ',\
                stt 
            cost_old = cost_new 
    return (x_new, it)

def lasso_fista(Y, D, Xinit, alambda, opts):
    """
    * Syntax: `[X, iter] = lasso_fista(Y, D, Xinit, lambda, opts)`
    * Solving a Lasso problem using FISTA [[11]](#fn_fista): 
        `X, = arg min_X 0.5*||Y - DX||_F^2 + lambda||X||_1`. 
        Note that `lambda` can be either a positive scalar or a matrix with 
        positive elements.
      - INPUT:
        + `Y, D, lambda`: as in the problem.
        + `Xinit`: Initial guess 
        + `opts`: options. See also [`fista`](#fista)
      - OUTPUT:
        + `X`: solution.
        + `iter`: number of fistat iterations.
    * **Note**:
      - _To see a toy example, un this function without inputs_
      - _Can be used for solving a Weighted Lasso problem_.
    """
    # pass 
    if Xinit.size == 0:
        Xinit = np.zeros((D.shape[1], Y.shape[1]))

    def calc_f(X):
        return 0.5*normF2(Y - np.dot(D, X))

    def calc_F(X):
        if isinstance(alambda, np.ndarray):
            return calc_f(X) + alambda*abs(X) # element-wise multiplication 
        else:
            return calc_f(X) + alambda*norm1(X)

    DtD = np.dot(D.T, D)
    DtY = np.dot(D.T, Y)
    def grad(X):
        g =  np.dot(DtD, X) - DtY 
        return g 
    L = np.max(LA.eig(DtD)[0])
    (X, it) = fista(grad, Xinit, L, alambda, opts, calc_F)
    return (X, it)

def lasso_fista_test():
    d = 300
    N = 700
    k = 700 
    Y = normc(np.random.rand(d, N))
    D = normc(np.random.rand(d, k))
    opts = Opts(test_mode = False, max_iter=50, show_cost = False)
    alambda = 0.01 
    lasso_fista(Y, D, np.array([]), alambda, opts)

class Opts:
    """
    parameters options. Store regularization parameters and algorithm stop 
    criteria
    """ 
    def __init__(self, tol = 1e-8, max_iter = 100, show_cost = False,\
        test_mode = False, lambda1 = None, lambda2 = None, lambda3 = None, \
        eta = None, check_grad = False, verbal = False):
        self.tol        = tol 
        self.max_iter   = max_iter 
        self.show_cost  = show_cost 
        self.test_mode  = test_mode
        self.check_grad = check_grad
        self.lambda1    = lambda1 
        self.lambda2    = lambda2 
        self.lambda3    = lambda3 
        self.verbal     = verbal

    def copy(self):
        opts = Opts()
        opts.tol        = self.tol 
        opts.max_iter   = self.max_iter 
        opts.show_cost  = self.show_cost 
        opts.test_mode  = self.test_mode
        opts.check_grad = self.check_grad
        opts.lambda1    = self.lambda1 
        opts.lambda2    = self.lambda2 
        opts.lambda3    = self.lambda3 
        return opts

class MyForm:
    """
    Describe a special family of matrices:
    A = [   M 0 ... 0;
            0 M ... 0;
            0 0 ... M] +
        [   N N ... N;
            N N ... N;
            N N ... N]
    with `k` block rows and columns 
    """
    def __init__(self, M, N, k):
        self.M = M.copy()
        self.N = N.copy()
        self.k = k 

    def full_express(self):
        return np.kron(np.eye(self.k), self.M) + \
            np.tile(self.N, (self.k, self.k))

    def mult(self, other):
        """
        Multiplication with other MyForm matrix 
        """
        A = np.dot(self.M, other.M)
        B = np.dot(self.M, other.N) + np.dot(self.N, other.M) + \
            self.k*np.dot(self.N, other.N)
        return MyForm(A, B, self.k)

    def inv(self):
        A = LA.inv(self.M)
        B = -np.dot(LA.inv(self.M + self.k*self.N), np.dot(self.N, A))
        return MyForm(A, B, self.k)

    def mult_vec(self, x):
        # X = np.reshape(x, (self.k, self.M.shape[1])).T
        X = myreshape(x, self.M.shape[1], self.k)
        p = np.dot(self.N, X).sum(axis = 1)
        # print self.M,'\n', X, '\n', self.M.dot(X)

        return vec(np.dot(self.M, X)) + vec(np.tile(p, (1, self.k)))


def MyForm_test():
    print('-------------------------------------------')
    print '`MyForm` test:'
    d = 20
    k = 50
    M = np.random.randint(10, size = (d, d))
    N = np.random.randint(10, size = (d, d))
    N = np.zeros_like(M)
    # M = np.zeros_like(N)
    # print N 
    A = np.random.randint(10, size = (d, d))
    B = np.random.randint(10, size = (d, d))

    P = MyForm(M, N, k)
    Q = MyForm(A, B, k)

    # Multiplication test 
    print '1. Multiplication test...',
    dif = LA.norm(np.dot(Q.full_express(),P.full_express()) - \
        Q.mult(P).full_express())
    if dif < 1e-8: 
        print 'diff =', dif, '\n   ...PASS'
    else:
        print 'diff =', dif, '\n   ...FAIL'

    # Inverse test 
    print '2. Inverse test ...',
    X = LA.inv(P.full_express())
    Y = P.inv()
    dif = LA.norm(X - Y.full_express())
    if dif < 1e-8: 
        print 'diff =', dif, '\n   ...PASS'
    else:
        print 'diff =', dif, '\n   ...FAIL'

    # vector multiplication 
    print '3. Multiplication with vector...',
    x = np.random.randint(3, size =(d*k,))
    # print x 
    # print P.full_express()
    y = np.dot(P.full_express(), x)
    z = P.mult_vec(x)

    # print 'true\n', y 
    # print 'computed\n', z 
    dif = LA.norm(y - z) 
    if dif < 1e-8: 
        print 'diff =', dif, '\n   ...PASS'
    else:
        print 'diff =', dif, '\n   ...FAIL'


def inv_IpXY(X, Y):
    """
    Calculating inv(I + XY)
    if X.shape[0] < X.shape[1], use regular inversion
    else 
    using the equality:
    inv(I + XY) = I - X*inv(I - YX)*Y 
    """
    if X.shape[0] < X.shape[1]:
        return LA.inv(np.eye(X.shape[0]) + np.dot(X, Y))
    else:
        return np.eye(X.shape[0]) - \
        X.dot(LA.inv(np.eye(X.shape[1]) + Y.dot(X))).dot(Y)

def inv_IpXY_test():
    print ('----------------------------------------')
    print '`inv_IpXY` test ...',
    d1 = 200
    d2 = 20 
    X = np.random.rand(d1, d2)
    Y = np.random.rand(d2, d1)
    
    t1 = time.time()
    M1 = LA.inv(np.eye(d1) + X.dot(Y))
    t2 = time.time()

    t3 = time.time()
    M2 = inv_IpXY(X, Y)
    t4 = time.time() 

    print '\nCompare running time:\n  regular: %5.2f' % (t2 - t1), \
        '(s)\n  fast   : %5.2f' % (t4 - t3), '(s)'
    dif = LA.norm(M1 - M2)
    if dif < 1e-8: 
        print 'diff =', dif, '\n   ...PASS'
    else:
        print 'diff =', dif, '\n   ...FAIL'

def num_grad(func, X):
    """
    Calculating gradient of a function `func(X)` where `X` is a matrix or 
    vector
    """
    grad = np.zeros_like(X)
    eps = 1e-4
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            # print X, '\n'
            Xp = X.copy()
            Xm = X.copy()
            Xp[i,j] += eps
            # print X 
            fp = func(Xp)
            Xm[i,j] -= eps
            fm = func(Xm)
            grad[i,j] = (fp - fm)/(2*eps)
    return grad 

def check_grad_test():
    """
    Test 'num_grad' and `check_grad`
    """
    print '----------------------------------'
    print 'Test num_grad ...',
    d = 20
    n = 20
    k = 20
    Y = np.random.rand(d, n)
    D = np.random.rand(d, k)
    X = np.random.rand(k, n) 
    def func(X):
        return 0.5*normF2(Y - D.dot(X)) 

    def grad_fnc(X):
        return np.dot(np.dot(D.T, D), X) - D.T.dot(Y) 

    def func2(X):
        return normF2(X)

    def grad_fnc2(X):
        return 2*X 

    print 'PASS' if check_grad(func, grad_fnc, X) else 'FAIL'
    print 'PASS' if check_grad(func2, grad_fnc2, X) else 'FAIL'

def check_grad(func, grad, X):
    print 'Checking grad...',
    grad1 = grad(X)
    grad2 = num_grad(func, X) 

    dif =  LA.norm(grad1 - grad2)
    if dif < 1e-5:
        print 'Different = %f' %dif, 'PASS'
    else:
        print 'Different = %f' %dif, 'FAIL'
    return dif < 1e-5

def randperm(n):
    return np.random.permutation(xrange(n))

def randperm_test():
    n = 10 
    print randperm(n)

def get_range(arange, c):
    return xrange(arange[c], arange[c+1])

def pickDfromY(Y, Y_range, D_range):
    """
    randomly pick k_c samples from Y_c 
    """ 
    C = Y_range.size - 1 
    D = np.zeros((Y.shape[0], D_range[-1]))
    for c in xrange(C):
        Yc = get_block_col(Y, c, Y_range) 
        N_c = Yc.shape[1]
        # print Yc 
        ids = randperm(N_c)
        range_Dc = get_range(D_range, c)
        kc = D_range[c+1] - D_range[c]
        D[:, D_range[c]:D_range[c+1]] = Yc[:, np.sort(ids[:kc])]
    return D 

def pickDfromY_test():
    print ('----------------------------------')
    print 'Test `pickDfromY`......'
    d = 2 
    n = 10 
    Y = np.random.randint(10, size=(d, n))
    print Y 
    Y_range = np.array([0, 4, 10])
    D_range = np.array([0, 2, 5])
    D = pickDfromY(Y, Y_range, D_range)
    # print Y_range 
    print D 

def myload(filename):
    # import testPickle 
    # A = sio.loadmat(filename)
    with open(filename, 'rb') as input_file:
    #     A = cPickle.load(input_file)
        A = pickle.load(input_file)
    return A 
    # return testPickle.pickle_load(filename)
    # print A['label_test']

def SRC_pred(Y, D, D_range, lambda1, opts):
    """
    * Classification based on SRC.
    * Syntax: `[pred, X] = SRC_pred(Y, D, D_range, lambda1, opts)`
      - INPUT:
        + `Y`: test samples.
        + `D`: the total dictionary. `D = [D_1, D_2, ..., D_C]` with `D_c` 
            being the _c-th_ class-specific dictionary.
        + `D_range`: range of class-specific dictionaries in `D`. 
        + `opts`: options.
          * `opts.lambda`: `lambda` for the Lasso problem.
          * `opts.max_iter`: maximum iterations of fista algorithm. 
          * others.
      - OUTPUT:
        + `pred`: predicted labels of test samples.
        + `X`: solution of the lasso problem.
    Ref:
    1. Wright, John, et al. "Robust face recognition via sparse representation."
       Pattern Analysis and Machine Intelligence, IEEE Transactions on, (2009)
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 4/6/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """
    C = D_range.size - 1 
    pred = np.zeros((1, Y.shape[1]))

    print "sparse coding...",
    X,_ = lasso_fista(Y, D, None, lambda1, opts)
    print "done"

    E = np.zeros((C, Y.shape[1]))
    for i in xrange(C):

        Xi = get_block_row(X, i, D_range)
        Di = get_block_col(D, i, D_range)
        R = Y - np.dot(Di, Xi)

        E[i,:] = (R*R).sum(axis = 0)

    pred = np.argmin(E, axis = 0) + 1

    return pred 

def pickTrainTest(dataset, N_train_c):
    data_fn = os.path.join('data', dataset + '.pickle')
    # data_fn = os.path.join('data', dataset + '.mat')
    Vars = myload(data_fn)
    print Vars.keys()
    Y = Vars['Y']
    d = Y.shape[0]
    if 'Y_range' not in Vars:
        Y_range = label_to_range(Vars['label'].flatten(1)).astype(int)
        
    else:
        Y_range = Vars['Y_range'].flatten(1).astype(int)

    C = Y_range.size - 1 
    N_total     = Y_range[-1]
    N_train     = C*N_train_c 
    N_test      = N_total - N_train 
    
    Y_train     = np.zeros((d, N_train))
    Y_test      = np.zeros((d, N_test))
    label_train = np.zeros((1, N_train))
    label_test  = np.zeros((1, N_test))    
    cur_train   = 0 
    cur_test    = 0 
    for c in xrange(C):
        Yc        = get_block_col(Y, c, Y_range) 
        N_total_c = Yc.shape[1] 
        N_test_c  = N_total_c - N_train_c 
        label_train[:, cur_train: cur_train + N_train_c] = \
            (c+1)*np.ones((1, N_train_c))
        label_test[:, cur_test:cur_test + N_test_c] = (c+1)*np.ones((1, N_test_c))

        ids = randperm(N_total_c)

        Y_train[:, cur_train: cur_train + N_train_c] = \
            Yc[:, np.sort(ids[:N_train_c])]

        Y_test[:, cur_test: cur_test + N_test_c] = \
            Yc[:, np.sort(ids[N_train_c:])]

        cur_train += N_train_c 
        cur_test += N_test_c 
    
    Y_train = normc(Y_train)
    Y_test  = normc(Y_test)
    return (Y_train, label_train, Y_test, label_test)

def range_reduce(D_range, bad_ids):
    C = D_range.size - 1 
    for c in xrange(C):
        cumk = D_range[c+1]
        e = cumk - np.nonzero(bad_ids < cumk)[0].size 
        D_range[c+1] = e 

def range_reduce_test():
    D_range = np.array([0, 4, 8, 13])
    bad_ids = np.array([1, 5, 7, 9, 10])
    print D_range, bad_ids
    range_reduce(D_range, bad_ids)
    print D_range 

# range_reduce_test()
def build_mean_vector(X, Y_range):
    """
    M = build_mean_vector(X, Y_range)
    suppose X = [X_1 X_2 ... X_C]
    """
    C = Y_range.size -1 
    # import pdb; pdb.set_trace()  # breakpoint 750a8d83 //
    M = np.zeros((X.shape[0], C)) 
    for c in xrange(C):
        Xc = get_block_col(X, c, Y_range)
        M[:, c] = np.mean(Xc, axis=1)
    return M 

def train_test_split(dataset, N_train):
    if dataset == 'myARgender':
        fn = os.path.join('data', 'myARgender.pickle')
        Vars = myload(fn)
        Y_train     = Vars['Y_train']
        Y_test      = Vars['Y_test']
        label_train = vec(Vars['label_train']).astype(int)
        label_test  = vec(Vars['label_test']).astype(int)
        range_train = label_to_range(label_train)
        range_test  = label_to_range(label_test)

        new_range_train = N_train * np.arange(N_train + 1)
        Y_train         = pickDfromY(Y_train, range_train, new_range_train)
        label_train     = range_to_label(new_range_train)

        Y_train = normc(Y_train)
        Y_test  = normc(Y_test)

    elif dataset == 'myARreduce':
        fn = os.path.join('data', 'AR_EigenFace.pickle')
        Vars = myload(fn)

        Y_train     = normc(Vars['tr_dat'])
        Y_test      = normc(Vars['tt_dat'])
        label_train = vec(Vars['trls']).astype(int)
        label_test  = vec(Vars['ttls']).astype(int)

    elif dataset == 'myFlower':
        dataset = 'myFlower102'
        fn      = os.path.join('data', dataset + '.pickle')
        Vars    = myload(fn)

        Y_train     = Vars['Y_train']
        Y_test      = Vars['Y_test']
        label_train = vec(Vars['label_train'])
        label_test  = vec(Vars['label_test'])
        range_train = label_to_range(label_train)
        range_test  = label_to_range(label_test)
        C = range_train.size - 1
        new_range_train = N_train * np.arange(C + 1)
        label_train     = range_to_label(new_range_train)
        Y_train         = pickDfromY(Y_train, range_train, new_range_train)

        Y_train = normc(Y_train)
        Y_test  = normc(Y_test)

    else: 
        Y_train, label_train, Y_test, label_test = \
            pickTrainTest(dataset, N_train)
    return (dataset, Y_train, Y_test, \
            label_train.astype(int), label_test.astype(int))

def SRC_top(dataset, N_train, lambda1):
    dataset, Y_train, Y_test, label_train, label_test = \
        train_test_split(dataset, N_train)

    opts = Opts(max_iter = 500, show_cost = True, test_mode = False)
    train_range = label_to_range(label_train)

    pred = SRC_pred(Y_test, Y_train, train_range, lambda1, opts)
    # print pred 
    acc = float(sum(pred == label_test))/label_test.size 
    return acc 

def build_mean_matrix(X, cols = None):
    m = np.mean(X, axis = 1)
    if cols == None: 
        return repmat(m, 1, X.shape[1]) 
    else:
        return np.tile(m, 1, cols)

def build_mean_matrix_test():
    print '----------------------------------'
    print '`build_mean_matrix` test: '
    X = np.random.randint(5, size = (2, 3))
    M = build_mean_matrix(X) 
    print X , '\n', M 

# build_mean_matrix_test()




#################### DLSI ##########################

#################### DFDL ##########################

#################### DLCOPAR #######################


def buildMhat(M, range_row, range_col):
    """
    buildMhat(M, range_row, range_col):
    suppose M = [M11 M12 ... M1n;
                      M21 M22 ... M3n;
                      .....
                      Mn1 Mn2 .... Mnn]
        then Mhat = = [2*M11  M12     ... M1n;
                       M21    2*M22   ... M3n;
                          .....
                       Mn1     Mn2 .... 2*Mnn]
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/21/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """
    C = range_row.size - 1 
    M2 = M.copy()
    for c in xrange(C):
        M2[range_row[c]: range_row[c+1], range_col[c]: range_col[c+1]] *= 2 
    return M2  

def buildMhat_test():
    print '------------------------------------------'
    print '`buildMhat` test: '
    d1 = 2
    d2 = 3 
    C = 2 
    range_row = d1*np.arange(C+1)
    range_col = d2*np.arange(C+1) 
    M = np.random.randint(3, size = (C*d1, C*d2))
    print 'M =\n', M 
    print 'buildMhat = \n', buildMhat(M, range_row, range_col)

def buildM_2Mbar(X, Y_range, lambda2):
    """
    """ 
    MM = np.zeros_like(X) 
    C = Y_range.size - 1 
    m = np.mean(X, axis = 1)
    for c in xrange(C):
        Xc = get_block_col(X, c, Y_range)
        mc = np.mean(Xc, axis = 1) 
        MM[:, Y_range[c]: Y_range[c+1]] = \
            repmat(lambda2*(m - 2*mc), 1, Y_range[c+1] - Y_range[c])
    return MM 

def buildMean(X):
    return build_mean_matrix(X)

def calc_acc(pred, ground_truth):
    acc = np.sum(pred == ground_truth)/ float(numel(ground_truth))
    return acc

def range_delete_ids(a_range, ids):
    """
    % function new_range = range_delete_ids(a_range, ids)
    % given a range `a_range` of an array. Suppose we want to delete some
    % element of that array indexed by `ids`, `new_range` is the new range
    """
    ids = np.sort(ids)
    n = numel(a_range)
    m = numel(ids)
    a = np.zeros_like(a_range)
    j = 1 
    while j < n-1:
        for i in xrange(n):
            while a_range[j] < ids[i]:
                j += 1
            for k in range(j, n):
                a[k] += 1

    new_range = a_range - a
    return new_range

def range_delete_ids_test():
    a_range = np.array([0, 3, 5, 10])
    ids = np.array([1, 4, 7, 10])
    print a_range
    print range_delete_ids(a_range, ids)

# range_delete_ids_test()

def max_eig(D):
    return np.max(LA.eig(D)[0])

def time_estimate(t):
    h = math.floor(t/3600)
    t -= 3600*h
    m = math.floor(t/60)
    t -= m*60
    print '| time left: %2dh%2dm%2ds' %(h, m, t)

def erase_diagonal(A):
    if A.shape[0] != A.shape[1]:
        print 'The input matrix is not square!'
        return 
    B = A.copy()
    np.fill_diagonal(B, 0)
    return B 

def erase_diagonal_blocks(A, row_range, col_range):
    if numel(row_range) != numel(col_range):
        print 'no. of column blocks != no. of row blocks!!'
    C = numel(row_range) - 1
    B = A.copy()
    for c in xrange(C):
        B[row_range[c]: row_range[c+1], col_range[c]: col_range[c+1]] = 0
    return B 

def inv_IpXY(X, Y):
    """
    Calculate the inverse of matrix A = I + XY.
    if X is a fat matrix (number of columns >> number of rows), then use inv(I + X*Y)
    else: use equation: (I + XY)^(-1) = I - Y*(I + Y*X)^(-1)*X 
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 4/12/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """
    d1 = X.shape[0]
    d2 = X.shape[1]
    if d1 > d2:
        M = eye(d1) - np.dot(np.dot(X, LA.inv(eye(d2) + np.dot(Y, X))), Y)
    else:
        M = LA.inv(eye(d1) + np.dot(X, Y))
    return M 
def inv_IpXY_test():
    d1 = 1000
    d2 = 10
    X = np.random.rand(d1, d2)
    Y = np.random.rand(d2, d1)
    t1 = time.time()
    A = LA.inv(eye(d1)+ np.dot(X,Y))
    t2 = time.time()
    print 't1 = ', t2 - t1 
    #
    t1 = time.time()
    B = inv_IpXY(X, Y)
    t2 = time.time()
    print 't2 = ', t2 - t1 
    print 'diff = ', normF2(A - B)

# inv_IpXY_test()

# test = False
# if test:
    # label_to_range_test()
    # range_to_label_test()
    # get_block_col_test()
    # get_block_row_test()
    # get_block_test()
    # vec_test()

    # norm1_test()
    # normF2_test()
    # # nuclearnorm_test()
    # # shrinkage_test()
    # # shrinkage_rank_test()
    # MyForm_test() 
    # inv_IpXY_test()
    # check_grad_test()
    # randperm_test()
    # pickDfromY_test()
    # myload('data/myARgender.pickle')
    # normF2_test()
    # t1 = time.time()
    # # acc = SRC_top('myARreduce', 7, 0.001)
    # t2 = time.time();
    # print acc, t2 - t1 
    # lasso_fista_test()
    # print 'hello'
    # ODL_test()
    # check_grad_test()
    # LRSDL_test()
    # buildMhat_test()
    # LRSDL_top('myYaleB', 50, 20, 5, 0.001, 0.01, .1)
    # min_rank_dict0_test()
    # repmat_test()
    # build_mean_matrix_test()
    # opts = Opts()
    # opts2 = opts 
    # print opts.max_iter 
    # opts2.max_iter = 0
    # print opts.max_iter 