import numpy as np 
from numpy import linalg as LA
# import scipy.io as sio
import cPickle
import math 
import time 
from time import strftime
import os 
import io 
# import theano.tensor as T
# from theano import function

# x = T.dmatrix('x')
# y = T.dmatrix('y')
# z = T.dot(x, y)
# mydot = function([x, y], z)






# TODO
test = True 
# test = False 

def repmat(A, rows, cols):
    """

    :param A:
    :param rows:
    :param cols:
    :return:
    """
    return np.tile(A, (cols, rows)).T

def repmat_test():
    print '------------------------------------\n'
    print 'Test `repmat`:'
    a = np.array([1, 2])
    rows = 2
    cols = 3 
    A = repmat(a, rows, cols)
    print 'a = ', a 
    print 'rows = %d,' %rows, 'cols = %d' %cols 
    print A 

def get_time_str():
    print 'Time now: ' + strftime("%m/%d/%Y %H:%M:%S")
    return strftime("%m%d_%H%M%S")

# print get_time_str()


def numel(A):
    """
    return number of elements of a numpy array 
    """
    return A.size 

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
    return M[:, col_range[c]: col_range[c+1]].copy()
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


def vec(A):
    """
    * Syntax: `a = vec(A)`
    * Vectorization of a matrix. This function is a built-in function in some 
    recent MATLAB version.
    """
    # pass
    # return A.reshape((-1, 1), order = 'F')
    return A.flatten(1)
    # x.reshape(c, r, order = 'F')
    # return np.reshape(A.flatten('F'), A.size, , order = 'F')

def vec_test():
    print('---------------------------------------')
    print('`vector` test:')
    A = np.random.randint(5, size = (3, 3))
    print "A = \n", A
    print "vec(A) = \n", vec(A) , vec(A).shape

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


def nuclearnorm_test():
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
    # pass 
    return np.maximum(0, U - alambda) + np.minimum(0, U + alambda)
    # np.maximum()

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

def min_rank_dict0(Y, X, lambdaD, Dinit, opts):
    """
    This function try to solve the following problem:
    [D] = argmin_D 0.5*|| Y - DX||_F^2 + lambdaD ||D||_*
    s.t. ||d_i||_2^2 <= 1, for all i 
    using ADMM:
    INPUT: 
        Y: Data 
        Dinit: intitial D 
        X: sparse code  
        lambdaD: regularization term 
    OUTPUT: 
        D: 
    Created: Tiep Vu 6/29/2015 2:05:28 PM
    ------------------------
    Choose a rho.
    Algorithm summary
    ADMM: D,J = argmin_DJ 0.5*||Y - DX||_F^2 + lambdaD||J||_*
    s.t ||d_i||_2^2 <= 1 and J = D
    Alternatively solving:
    (1): D^{k+1} = argmin_D 0.5*||Y - DX||_F^2 + rho/2 ||J - D + U^k||_F^2 s.t. ||d_i||_2^2 <= 1
        this problem can be soved using the update dictionary stage in Online Dictionary Learning method
    (2): J^{k+1} = argminJ lambdaD||J||_* + rho/2||J - D^{k+1} + U^k||
        Solution: shrinkage_rank(D^{k+1} - U^k, lambdaD/rho)
    (3): Update U: U^{k+1} = U^k + J^{k+1} - D^{k+1}

    Stoping cretia:
    ||r^k||_F^2 <= tol, ||s^k||_F^2 <= tol 
    r^k = J^k - D^k 
    s^k = rho(J^{k+1} - J^k) 
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/22/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """
    YXt = np.dot(Y, X.T)
    XXt = np.dot(X, X.T)
    rho = 0.25 
    D_old = Dinit 
    J_old = Dinit 
    U_old = np.zeros_like(Dinit)
    it = 0 
    I = np.eye(XXt.shape[0])
    tau = 2 
    mu = 10.0 
    optsD = opts 
    optsD.max_iter = 50
    while it < opts.max_iter:
        it += 1 
        ## =========update D ================================
        # D = argmin_D 0.5*||Y - DX||_F^2 + rho/2 ||J - D + U||_F^2 s.t. ||d_i||_2^2 <= 1
        E = YXt + rho*(J_old + U_old)
        F = XXt + rho*I
        # D_new = updateD_EF(D_old, E, F, 10);
        D_new = ODL_updateD(D_old, E, F, optsD)[0]
        ## ========= update J ==============================
        # J^{k+1} = argminJ lambdaD||J||_* + rho/2||J - D + U||
        J_new = np.real(shrinkage_rank(D_new - U_old, lambdaD/rho))

        ## ========= update U ==============================
        U_new = U_old + J_new - D_new
        
        ## ========= check stop ==============================
        r = J_new - D_new
        s = rho*(J_new - J_old)
        r_eps = LA.norm(r, 'fro')
        s_eps = LA.norm(s, 'fro')   
        if r_eps < opts.tol and s_eps < opts.tol:
            break
        D_old = D_new
        J_old = J_new
        U_old = U_new
        if r_eps > mu*s_eps:
            rho = rho*tau
        elif s_eps > mu*r_eps:
            rho = rho/tau

    return D_new 

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
        if opts.test_mode:
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
    # DtD = mydot(D.T, D)
    DtY = np.dot(D.T, Y)
    # DtY = mydot(D.T, Y)
    def grad(X):
        # t1 = time.time() 

        g=  np.dot(DtD, X) - DtY 
        # g = mydot(DtD, X) - DtY 
        # t2 = time.time() 
        # print t2 - t1 
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
    # opts = {tol : 0.00001, max_iter: 30}
    opts = Opts(test_mode = False, max_iter=50, show_cost = False)
    alambda = 0.01 
    lasso_fista(Y, D, np.array([]), alambda, opts)
    # pass 

class Opts:
    """
    parameters options. Store regularization parameters and algorithm stop 
    criteria
    """ 
    def __init__(self, tol = 1e-8, max_iter = 100, show_cost = False,\
        test_mode = False, lambda1 = None, lambda2 = None, lambda3 = None, \
        eta = None, check_grad = False):
        self.tol        = tol 
        self.max_iter   = max_iter 
        self.show_cost  = show_cost 
        self.test_mode  = test_mode
        self.check_grad = check_grad
        self.lambda1    = lambda1 
        self.lambda2    = lambda2 
        self.lambda3    = lambda3 

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
    # A = sio.loadmat(filename)
    with open(r""+filename, "rb") as input_file:
        A = cPickle.load(input_file)
    return A 
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


# def picktrntst(Y, Y_range, N_trc_c):
#   """
#   Description : pick training set and test set from data
#       INPUT:
#       Y: data (each column is an observation)
#       label: label of data (start from 1 to C - nmber of class)
#       N_trn_c: number of training samples in each class
#       OUTPUT: 
#       Y_trn: picked training data 
#       label_trn: label of training data 
#       Y_tst: test data 
#       label_tst: test label
#   -----------------------------------------------
#   Author: Tiep Vu, thv102@psu.edu, 4/16/2016
#           (http://www.personal.psu.edu/thv102/)
#   -----------------------------------------------
#   """



#   return (Y_trn, label_trn, Y_tst, label_tst)

# def picktrntst_wrapper(dataset, N_trn):
#   """
#   Pick N_trn samples from each class in `dataset`
#   """ 
#   data_fn = os.path.join('data', dataset + '.mat')
#   Vars = myload(data_fn)
#   if 'Y_range' not in Vars:
#       Y_range = label_to_range(Vars['label'])
#   else:
#       Y_range = Vars['Y_range']
#   Y = normc(Vars['Y'])
#   return picktrntst(Y, Y_range, N_trn)
    # pass
    # print D_range 
##############test 
def pickTrainTest(dataset, N_train_c):
    data_fn = os.path.join('data', dataset + '.pickle')
    Vars = myload(data_fn)
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
    
    Y_train     = np.zeros((d, N_train) )
    Y_test      = np.zeros((d, N_test))
    label_train = np.zeros((1, N_train))
    label_test  = np.zeros((1, N_test)
)    
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

def zeros(rows, cols):
    return np.zeros((rows, cols))

def ones(rows, cols):
    return np.ones((rows, cols))
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

#################### ODL ###########################
def ODL_cost(Y, D, X, alambda):

    return 0.5*normF2(Y - np.dot(D, X)) + alambda*norm1(X)

def ODL_updateD(D, E, F, opts):
    """
    * Syntax `[D, it] = ODL_updateD(D, E, F, opts)`
    * The main algorithm in ODL. 
    * Solving the optimization problem:
      `D = arg min_D -2trace(E'*D) + trace(D*F*D')` 
        subject to: `||d_i||_2 <= 1`,
        where `F` is a positive semidefinite matrix. 
      - INPUT: 
        + `D, E, F` as in the above problem.
        + `opts`. options:
          * `opts.max_iter`: maximum number of iterations.
          * `opts.tol`: when the difference between `D` in two successive 
            iterations less than this value, the algorithm will stop.
      - OUTPUT:
        + `D`: solution.
        + `it`: number of run iterations.
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/07/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """ 
    Dold = D 
    it = 0 
    sizeD = D.size 
    while it < opts.max_iter:
        it += 1
        for i in xrange(D.shape[1]):
            if F[i,i] != 0:
                a = 1./F[i,i] * (E[:, i] - np.dot(D, F[:, i])) + D[:, i]
                D[:, i] = a/ max(LA.norm(a), 1)
        if LA.norm(D - Dold, 'fro')/sizeD < opts.tol:
            break
        Dold = D 
    return (D, it)
    # pass 

def ODL(Y, k, alambda, opts):
    """
    * An implementation of the well-known Online Dictionary Learning method 
    * Syntax: (D, X) = ODL(Y, k, opts)
    * Solving the dictionary learning problem:
       `[D, X] = arg min_{D, X} 0.5||Y - DX||_F^2 + lambda||X||_1` 
        subject to `||d_i||_2 <= 1`.
    ----------------------------
    * Ref:
    Mairal, Julien, et al. "Online learning for matrix factorization and 
    sparse coding." _The Journal of Machine Learning Research 11_ (2010): 
    [[paper]](http://www.di.ens.fr/~fbach/mairal10a.pdf)
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/20/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """ 
    if opts.show_cost:
        print 'Start ODL...'
    D = pickDfromY(Y, np.array([0, Y.shape[1]]), np.array([0, k]))

    X = np.zeros((D.shape[1], Y.shape[1]))
    if opts.show_cost:
        print 'cost init = ', ODL_cost(Y, D, X, alambda)
    ## 
    optsD = opts 
    optsD.max_iter = 200 
    optsX = opts 
    optsX.max_iter = 300
    it = 0 
    cost_old = float("inf")
    while it < opts.max_iter:
        it += 1 
        ## Sparse coding step 
        X,_ = lasso_fista(Y, D, X, alambda, optsX)
        if opts.show_cost:
            cost_new = ODL_cost(Y, D, X, alambda)
            if cost_new < cost_old:
                stt = 'YES'
            else:
                stt = 'NO, cost increases!!!'
            print 'iter = %3d' %it, '| costX = %4.4f |' % cost_new, \
                'cost decreases?', stt 
            cost_old = cost_new 
        ## Dictionary update step 
        E = np.dot(Y, X.T)
        F = np.dot(X, X.T)
        D,_ = ODL_updateD(D, E, F, optsD)
        if opts.show_cost:
            cost_new = ODL_cost(Y, D, X, alambda)
            if cost_new < cost_old:
                stt = 'YES'
            else:
                stt = 'NO, cost increases!!!'
            print 'iter = %3d' %it, '| costD = %4.4f |' % cost_new, \
                'cost decreases?', stt  
            cost_old = cost_new
    return (D, X)   

def ODL_test():
    d = 10 
    N = 100 
    k = 50 
    alambda = 0.1 
    Y = normc(np.random.rand(d, N))
    opts = Opts(max_iter = 100, show_cost = True)
    ODL(Y, k, alambda, opts)



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

    # pass 
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

#################### DLSI ##########################
def DLSI_term(D, D_range):
    """
    cost =  DLSI_term(D, D_range):
    """ 
    cost = 0 
    C = D_range.size - 1 
    for c in range(C):
        ranged = np.arange(D_range[c], D_range[c+1])
        rangecomd = np.setdiff1d(np.arange(D_range[-1]), ranged)
        Dc = D[:, ranged]
        D_comc = D[:, range_comd]
        cost += normF2(np.dot(D_comc.T, Dc))
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
    cost = 0.5*opts.eta *DLSI_term(D, D_range)
    
    for c in xrange(C):
        Yc = get_block_col(Y, c, Y_range)
        Xc = X[c]
        Dc = get_block_col(D, c, Y_range)
        cost += .5*normF2(Yc - np.dot(Dc, Xc)) + norm1(Xc)

    return cost 
    
def DLSI_updateX(Y, Y_range, D, D_range, X, lambda1):
    pass 

def DLSI_updateD(Y, Y_range, D, D_range, X, eta):
    pass 

def DLSI_pred(Y, Y_label, D, D_range, lambda1, eta):
    pass 

def DLSI_top(dataset, n_c, k, alambda, eta):
    """
    DLSI_top(dataset, n_c, k, alambda, eta)
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/19/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """
    print "Apply DLSI on " + dataset + "with parameters:"
    print "n_c: ", n_c, '\nk: ', k, '\nlambda: ',\
            alambda, '\neta: ', eta
    ## get data
    print "\nPreparing training and test samples...",
    dataset, Y_train, Y_test, label_train, label_test = \
        train_test_split(dataset, n_c)
    print "done"
    ## output filename 
    path = 'results/DLSI'
    if not os.path.exists(path):
        os.makedirs(path)
    t = get_time_str()
    fn = os.path.join('results', 'DLSI', dataset + '_N_'+ str(n_c) + \
        '_k_' + str(k) + '_l_' + str(lambda) + '_e_' + str(eta) + '_' + t \
        + '.pickle')
    output_file = open(fn, 'w+')
    ## Prepare parameters
    C = np.unique(label_train).size 
    opts = Opts(max_iter = 100, \
                alambda  = alambda, \
                eta  = eta)
    D_range     = k*np.arange(C+1)
    Y_range     = label_to_range(label_train)
    ## Train 
    print "Training ...",
    D, X = DLSI(Y_train, Y_range, D_range, opts)
    print "...done training"
    ##
    print "Test...",
    acc = DLSI_pred(Y_test, D,  opts)
    print "...done test"
    ## save results
    A = {'acc': acc}
    cPickle.dump(A, output_file)
    close(output_file)
#################### DFDL ##########################

#################### DLCOPAR #######################

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
           opts.lambda3*nuclearnorm(D0)
    return cost 

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
        MM[:, Y_range[c]: Y_range[c+1]] = repmat(lambda2*(m - 2*mc), \
                                        1, Y_range[c+1] - Y_range[c])
    return MM 

def buildMean(X):
    # N = X.shape[1]
    # m = np.mean(X, axis = 1) 
    # M = np.tile(m, (1, N))
    return build_mean_matrix(X)


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
    C = Y_range.size - 1 
    D = np.zeros((Y.shape[0], D_range_ext[-2]))
    X = np.zeros((D_range_ext[-2], Y.shape[1]))
    ## class-specific dictionary 
    optsODL = Opts(max_iter = 50, tol = 1e-8)
    for c in xrange(C):
        Yc = get_block_col(Y, c, Y_range)
        D[:, D_range_ext[c]:D_range_ext[c+1]], Xcc = ODL(Yc, D_range_ext[c+1] -\
            D_range_ext[c], opts.lambda1, optsODL)
        X[D_range_ext[c]:D_range_ext[c+1], Y_range[c]:Y_range[c+1]] = \
            Xcc.copy()
    ## shared dictionary 
    k0 = D_range_ext[-1] - D_range_ext[-2]
    if k0 > 0:
        D0, X0 = ODL(Y, k0, opts.lambda1, opts)
    return (D, D0, X, X0)


def LRSDL(Y, Y_range, D_range_ext, opts):
    """
    D, D_range, D0, X, X0, M, m0 = LRSDL(Y, Y_range, D_range_ext, opts)
    """ 
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
    



# test = False
if test:
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
    LRSDL_top('myFlower', 10, 8, 5, 0.01, 0.1, .1)
    # min_rank_dict0_test()
    # repmat_test()
    # build_mean_matrix_test()
    # opts = Opts()
    # opts2 = opts 
    # print opts.max_iter 
    # opts2.max_iter = 0
    # print opts.max_iter 