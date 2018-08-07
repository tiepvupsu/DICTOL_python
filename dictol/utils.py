import math
# import sys
import time
from time import strftime
import numpy as np
import numpy.linalg as LA
# from six.moves import cPickle as pickle
import os
# import io
import scipy.io as sio
import pkg_resources
# import pickle
# from ODL import *
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

def vec(A):
    """
    * Syntax: `a = vec(A)`
    * Vectorization of a matrix. This function is a built-in function in some
    recent MATLAB version.
    """
    # pass
    # return A.reshape((-1, 1), order = 'F')
    return A.flatten(1)

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
        label[arange[i]: arange[i+1]] *= (i+1)
    return label

def range_to_label_test():
    arange = np.array([0, 3, 5, 9])
    print "---------------------------------------\n`range_to_label` test:"
    print "range: ", arange
    print "label: ", range_to_label(arange)

def get_block_col(M, C, col_range):
    """
    * Syntax: `Mc = get_block_col(M, c, col_range)`
    * Extract a block of columns from a matrix.
        - `M`: the big matrix `M = [M_1, M_2, ...., M_C]`.
        - `C`: blocks indices (start at 0).
        - `col_range`: range of samples, see `Y_range` and `D_range` above.
    * Example: `M` has 25 columns and `col_range = [0, 10, 25]`, then
    `get_block_col(M, 1, col_range)` will output the first block of `M`,
    i.e. `M(:, 1:10)`.
    """
    if isinstance(C, int):
        return M[:, col_range[C]: col_range[C+1]]
    if isinstance(C, list) or isinstance(C, (np.ndarray, np.generic)):
        ids = []
        for c in C:
            ids = ids + range(col_range[c], col_range[c+1])
        return M[:, ids]




def get_block_col_test():
    print "---------------------------------------\n`get_block_col test:"
    A = np.random.randint(5, size=(3, 9))
    arange = np.array([0, 4, 7, 9], dtype = np.int)
    print " A: \n", A
    print " arange: ", arange
    print " get_block_col(A, [1, 2], arange): \n", get_block_col(A, [1, 2], arange)

def get_block_row(M, C, row_range):
    """
    * Extract a block of rows from a matrix.
    * Syntax: `Mc = get_block_row(M, c, row_range)`
        - `M`: the big matrix `M = [M_1; M_2; ....; M_C]`.
        - `C`: an `int`, `list` of ints or an nparray of ints
            block indices (start at 0).
        - `row_range`: range of samples, see `Y_range` and `D_range` above.
    * Example: `M` has 40 rows and `row_range = [0, 10, 25, 40]`, then
    `get_block_row(M, 2, row_range)` will output the second block of `M`,
    i.e. `M(11:25, :)`.
    """
    if isinstance(C, int):
        return M[row_range[C]: row_range[C+1], :].copy()
    if isinstance(C, list) or isinstance(C, (np.ndarray, np.generic)):
        ids = []
        for c in C:
            ids = ids + range(row_range[c], row_range[c+1])
        return M[ids, :].copy()

def get_block_row_test():
    print "---------------------------------------\n`get_block_row` test:"
    A = np.random.randint(5, size=(9, 3))
    arange = np.array([0, 4, 7, 9], dtype = np.int)
    print " A: \n", A
    print " arange: ", arange
    print " get_block_row(A, [0, 2], arange): \n", get_block_row(A, np.array([0, 2]), arange)


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
    if X.shape[0]*X.shape[1] == 0:
        return 0
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
    if X.shape[0]*X.shape[1] == 0:
        return 0
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
    """
    normalize each column of A to have norm2 = 1
    """
    return A/ np.tile(np.sqrt(np.sum(A*A, axis=0)), (A.shape[0], 1))

def nuclearnorm(X):
    """
    * Return nuclear norm of a matrix.
    * Syntax `res = nuclearnorm(X)`
    """
    if X.size == 0:
        return 0
    return LA.norm(X) if is_vector(X) else LA.norm(X, 'nuc')


    pass

def shrinkage(U, alambda):
    """
    * Soft thresholding function.
    * Syntax: `X = shrinkage(U, lambda)`
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
        # range_Dc = get_range(D_range, c)
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
    return sio.loadmat(filename)


def pickTrainTest(dataset, N_train_c):
    data_fn = pkg_resources.resource_filename('dictol', 'data/'+dataset + '.mat') 
    # data_fn = os.path.join('dictol/data', dataset + '.mat')
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
    return M = [m1, m2, ..., M_C]
    where mi = mean(X_i)
    """
    C = Y_range.size -1
    M = np.zeros((X.shape[0], C))
    for c in xrange(C):
        Xc = get_block_col(X, c, Y_range)
        M[:, c] = np.mean(Xc, axis=1)
    return M

def train_test_split(dataset, N_train):
    if dataset == 'myARgender':
        fn = pkg_resources.resource_filename('dictol', 'data/'+dataset + '.mat') 
        # fn = os.path.join('data', 'myARgender.pickle')
        Vars = myload(fn)
        Y_train     = Vars['Y_train']
        Y_test      = Vars['Y_test']
        label_train = vec(Vars['label_train']).astype(int)
        label_test  = vec(Vars['label_test']).astype(int)
        range_train = label_to_range(label_train)
        # range_test  = label_to_range(label_test)

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
        # range_test  = label_to_range(label_test)
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
    C = len(range_row) - 1
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

def build_mean_matrix(X, cols = None):
    """
    repeat np.mean(X, axis = 1) cols times.

    ---------------
    Parameters:
    X: 2d numpy array
    cols: int
        if cols == None then cols = #cols of X

    """
    if len(X.shape) < 2 or X.shape[1] == 0:
        return X
    m = np.mean(X, axis = 1)
    if cols == None:
        return repmat(m, 1, X.shape[1])
    else:
        return np.tile(m, 1, cols)

def buildMean(X):
    return build_mean_matrix(X)

def calc_acc(pred, ground_truth):
    acc = np.sum(pred == ground_truth)/ float(ground_truth.size)
    return acc

def range_delete_ids(a_range, ids):
    """
    % function new_range = range_delete_ids(a_range, ids)
    % given a range `a_range` of an array. Suppose we want to delete some
    % element of that array indexed by `ids`, `new_range` is the new range
    """
    ids = np.sort(ids)
    n = a_range.size
    # m = ids.size
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
    if len(row_range) != len(col_range):
        print 'no. of column blocks != no. of row blocks!!'
    C = len(row_range) - 1
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
        M = np.eye(d1) - np.dot(np.dot(X, LA.inv(np.eye(d2) + np.dot(Y, X))), Y)
    else:
        M = LA.inv(np.eye(d1) + np.dot(X, Y))
    return M

def inv_IpXY_test():
    d1 = 1000
    d2 = 10
    X = np.random.rand(d1, d2)
    Y = np.random.rand(d2, d1)
    t1 = time.time()
    A = LA.inv(np.eye(d1)+ np.dot(X,Y))
    t2 = time.time()
    print 't1 = ', t2 - t1
    #
    t1 = time.time()
    B = inv_IpXY(X, Y)
    t2 = time.time()
    print 't2 = ', t2 - t1
    print 'diff = ', normF2(A - B)

def progress_str(cur_val, max_val, total_point=50):
    p = int(math.ceil(float(cur_val)*total_point/ max_val))
    return '|' + p*'#'+ (total_point - p)*'.'+ '|'

