import os
import math
import time
from time import strftime
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import pkg_resources


def repmat(A, rows, cols):
    return np.tile(A, (cols, rows)).T


def vec(A):
    # TODO: rewrite docstrings
    """
    * Syntax: `a = vec(A)`
    * Vectorization of a matrix. This function is a built-in function in some
    recent MATLAB version.
    """
    return A.flatten(1)


def label_to_range(label):
    """
    Convert label to range

    Parameters:
    -----------
    label: list of integers
        must be in the form of [1, 1, ..., 1, 2, 2, ..., 2, ..., C, C, ..., C]
        i.e. nondecreasing numbers starting from 1, each element is greater
        than the previous element by at most 1

    Returns:
    --------
    a list of intergers with C + 1 elements, start with 0
    the i-th element is number of elements in label that equals to i
        
    """
    res = [0]
    assert label[0] == 1, 'label must start with 1'
    for i in range(1, len(label)):
        if label[i] == label[i-1]:
            continue
        if label[i] == label[i-1] + 1:
            res.append(i)
        else:
            assert False,\
                ('label[{}] and label[{}] must be equal or two consecutive '
                 'integers, got {} and {}').format(
                     i-1, i, label[i-1], label[i]
                 )
    res.append(len(label))
    return res


def range_to_label(a_range):
    """
    From a range, convert it to label

    This is an inverse function of label_to_range
    Parameters:
    -----------
    a_range: list of integers
        must start with 0 and is a strictly increasing list

    Returns:
    --------

    """
    assert a_range[0] == 0, 'input must start with 0'
    res = []
    for i in range(1, len(a_range)):
        assert a_range[i] > a_range[i-1],\
            ('a_range must be an increasing list, '
             'got a_range[{}] = {} < a_range[{}] = {}').format(
                 i, a_range[i], i - 1, a_range[i-1]
             )

        res.extend([i]*(a_range[i] - a_range[i-1]))
    return res


def get_block_row(matrix, block_indices, row_range):
    """
    Extract a subset of rows from a matrix

    Parameters:
    -----------
    matrix: 2-d numpy array
        block matrix
    block_indices: integer of list of integers
        indices of extracted blocks, 0-indexed. If indices is a list, return
        the concatenation of all blocks
    row_range: list of intergers
        in the form of [0, c_1, c_1 + c_2, ..., c_1 + c_2 + ... + c_N]
        where c_i is the number of rows in the i-th block

    Returns:
    --------
    a 2-d matrix
    """
    assert matrix.ndim == 2, 'Expect to receive 2-d array input, got shape {}'.format(matrix.shape)
    if isinstance(block_indices, int):
        block_indices = [block_indices]
    # if isinstance(block_indices, (list, np.ndarray, np.generic))
    ids = []
    for i in block_indices:
        ids = ids + list(range(row_range[i], row_range[i+1]))
    return matrix[ids, :].copy()


def get_block_col(matrix, block_indices, col_range):
    """
    Extract a subset of columns from a matrix

    Parameters:
    -----------
    matrix: 2-d numpy array
        block matrix
    block_indices: integer of list of integers
        indices of extracted blocks, 1-indexed. If indices is a list, return
        the concatenation of all blocks
    row_range: list of intergers
        in the form of [0, c_1, c_1 + c_2, ..., c_1 + c_2 + ... + c_N]
        where c_i is the number of columns in the i-th block

    Returns:
    --------
    a 2-d matrix
    """
    assert matrix.ndim == 2, 'Expect to receive 2-d array input, got shape {}'.format(matrix.shape)
    assert matrix.shape[1] == col_range[-1]
    return get_block_row(matrix.T, block_indices, col_range).T


def get_block(matrix, i, j, row_range, col_range):
    """
    Extract a submatrix of a matrix

    Parameters:
    -----------
    matrix the big matrix:
    matrix = [ M11, M12, ..., M1m;
               M21, M22, ..., M2m;
               ... ;
               Mn1, Mn2, ..., Mnm]
    i: row block index
    j: column block index
    row_range: row range
    col_range: columns range
    """
    return matrix[row_range[i]:row_range[i+1],
                  col_range[j]: col_range[j+1]].copy()


def norm1(X):
    """
    Return norm 1 of a matrix, which is sum of the absolute value of all elements
    of that matrix.
    """
    if X.shape[0]*X.shape[1] == 0:
        return 0
    return abs(X).sum()


def normF2(X):
    """
    Return square of the Frobenius norm, which is sum of square of all
    elements in a matrix
    """
    if X.shape[0]*X.shape[1] == 0:
        return 0
    return LA.norm(X, 'fro')**2


def normc(A):
    """
    normalize each column of A to have norm2 = 1
    """
    return A / np.tile(np.sqrt(np.sum(A*A, axis=0)), (A.shape[0], 1))


def nuclearnorm(X):
    """
    Return nuclear norm of a matrix.
    """
    if X.size == 0:
        return 0
    return LA.norm(X) if X.ndim == 1 else LA.norm(X, 'nuc')


def shrinkage(U, alambda):
    """
    Soft thresholding function.
    
    Solve the following optimization problem:
    X = arg min_X 0.5*||X - U||_F^2 + lambda||X||_1
    where U and X are matrices with same sizes. lambda can be either a positive
    scalar or a positive matrix (all elements are positive) with same size as X.
    """
    return np.maximum(0, U - alambda) + np.minimum(0, U + alambda)


def shrinkage_rank(D, alambda):
    """
    Singular value thresholding algorithm for matrix completion.
    Solve the following optimization problem:
      X = arg min_X 0.5*||X - D||_F^2 + lambda*||X||_*
      where ||X||_* is the nuclear norm.
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
    with k block rows and columns
    """

    def __init__(self, M, N, k):
        self.M = M.copy()
        self.N = N.copy()
        self.k = k

    def full_express(self):
        return np.kron(np.eye(self.k), self.M) + np.tile(self.N, (self.k, self.k))

    def mult(self, other):
        """
        Multiplication with other MyForm matrix
        """
        A = np.dot(self.M, other.M)
        B = np.dot(self.M, other.N) + np.dot(self.N, other.M) + \
            self.k*np.dot(self.N, other.N)
        return MyForm(A, B, self.k)

    def inv(self):
        """
        compute inverse matrix
        """
        A = LA.inv(self.M)
        B = - np.dot(LA.inv(self.M + self.k*self.N), np.dot(self.N, A))
        return MyForm(A, B, self.k)

    def mult_vec(self, x):
        """
        return M*x (matrix vector multiplication)
        """
        X = x.reshape(self.M.shape[1], self.k, order='F')
        p = np.dot(self.N, X).sum(axis=1)

        return vec(np.dot(self.M, X)) + vec(np.tile(p, (1, self.k)))


def randperm(n):
    """
    get a random permutation of range(n)
    """
    return np.random.permutation(list(range(n)))


# def get_range(arange, c):
#     return list(range(arange[c], arange[c+1]))


def pickDfromY(Y, Y_range, D_range):
    """
    randomly pick k_c samples from Y_c
    """
    C = Y_range.size - 1
    D = np.zeros((Y.shape[0], D_range[-1]))
    for c in range(C):
        Yc = get_block_col(Y, c, Y_range)
        N_c = Yc.shape[1]
        # print Yc
        ids = randperm(N_c)
        kc = D_range[c+1] - D_range[c]
        D[:, D_range[c]:D_range[c+1]] = Yc[:, np.sort(ids[:kc])]
    return D


def load_mat(filename):
    return sio.loadmat(filename)


def picl_train_test(dataset, N_train_c):
    data_fn = pkg_resources.resource_filename('dictol', 'data/'+dataset + '.mat') 
    vars_dict = load_mat(data_fn)
    Y = vars_dict['Y']
    d = Y.shape[0]
    if 'Y_range' not in vars_dict:
        Y_range = label_to_range(vars_dict['label'].flatten(1)).astype(int)

    else:
        Y_range = vars_dict['Y_range'].flatten(1).astype(int)

    C = Y_range.size - 1
    N_total     = Y_range[-1]
    N_train     = C*N_train_c
    N_test      = N_total - N_train

    Y_train     = np.zeros((d, N_train))
    Y_test      = np.zeros((d, N_test))
    label_train = [0]*N_train
    label_test = [0]*N_test
    cur_train   = 0
    cur_test    = 0
    for c in range(C):
        Yc        = get_block_col(Y, c, Y_range)
        N_total_c = Yc.shape[1]
        N_test_c  = N_total_c - N_train_c
        label_train[cur_train: cur_train + N_train_c] = [c+1]*N_train_c
        label_test[cur_test:cur_test + N_test_c] = [c+1]*N_test_c

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
    for c in range(C):
        cumk = D_range[c+1]
        e = cumk - np.nonzero(bad_ids < cumk)[0].size
        D_range[c+1] = e


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
    for c in range(C):
        Xc = get_block_col(X, c, Y_range)
        M[:, c] = np.mean(Xc, axis=1)
    return M


def train_test_split(dataset, N_train):
    if dataset == 'myARgender':
        fn = pkg_resources.resource_filename('dictol', 'data/'+dataset + '.mat') 
        # fn = os.path.join('data', 'myARgender.pickle')
        vars_dict = load_mat(fn)
        Y_train = vars_dict['Y_train']
        Y_test = vars_dict['Y_test']
        label_train = vec(vars_dict['label_train']).astype(int)
        label_test = vec(vars_dict['label_test']).astype(int)
        range_train = label_to_range(label_train)
        # range_test  = label_to_range(label_test)

        new_range_train = N_train * np.arange(N_train + 1)
        Y_train         = pickDfromY(Y_train, range_train, new_range_train)
        label_train     = range_to_label(new_range_train)

        Y_train = normc(Y_train)
        Y_test  = normc(Y_test)

    elif dataset == 'myARreduce':
        fn = os.path.join('data', 'AR_EigenFace.pickle')
        vars_dict = load_mat(fn)

        Y_train     = normc(vars_dict['tr_dat'])
        Y_test      = normc(vars_dict['tt_dat'])
        label_train = vec(vars_dict['trls']).astype(int)
        label_test  = vec(vars_dict['ttls']).astype(int)

    elif dataset == 'myFlower':
        dataset = 'myFlower102'
        fn      = os.path.join('data', dataset + '.pickle')
        vars_dict    = load_mat(fn)

        Y_train     = vars_dict['Y_train']
        Y_test      = vars_dict['Y_test']
        label_train = vec(vars_dict['label_train'])
        label_test  = vec(vars_dict['label_test'])
        range_train = label_to_range(label_train)
        num_classes = len(range_train) - 1
        new_range_train = N_train * np.arange(num_classes + 1)
        label_train = range_to_label(new_range_train)
        Y_train = pickDfromY(Y_train, range_train, new_range_train)

        Y_train = normc(Y_train)
        Y_test = normc(Y_test)

    else:
        Y_train, label_train, Y_test, label_test = picl_train_test(dataset, N_train)
    return (Y_train, Y_test, label_train, label_test)


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
    for c in range(C):
        M2[range_row[c]: range_row[c+1], range_col[c]: range_col[c+1]] *= 2
    return M2


def buildM_2Mbar(X, Y_range, lambda2):
    """
    """
    MM = np.zeros_like(X)
    C = len(Y_range) - 1
    m = np.mean(X, axis=1)
    for c in range(C):
        Xc = get_block_col(X, c, Y_range)
        mc = np.mean(Xc, axis=1)
        MM[:, Y_range[c]: Y_range[c+1]] = repmat(lambda2*(m - 2*mc), 1, Y_range[c+1] - Y_range[c])
    return MM


def build_mean_matrix(X, cols=None):
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
    mean_vector = np.mean(X, axis=1)
    if cols is None:
        return repmat(mean_vector, 1, X.shape[1])
    else:
        return np.tile(mean_vector, cols)


def range_delete_ids(a_range, ids):
    """
    given a range a_range of an array. Suppose we want to delete some
    element of that array indexed by `ids`, `new_range` is the new range
    """
    ids = np.sort(ids)
    n = a_range.size
    a = np.zeros_like(a_range)
    j = 1
    while j < n-1:
        for i in range(n):
            while a_range[j] < ids[i]:
                j += 1
            for k in range(j, n):
                a[k] += 1

    new_range = a_range - a
    return new_range


def range_delete_ids_test():
    a_range = np.array([0, 3, 5, 10])
    ids = np.array([1, 4, 7, 10])
    print(a_range)
    print(range_delete_ids(a_range, ids))


def max_eig(D):
    """
    return maximum eigenvalue of matrix D
    """
    return np.max(LA.eig(D)[0])


def erase_diagonal(A):
    if A.shape[0] != A.shape[1]:
        print('The input matrix is not square!')
        return
    B = A.copy()
    np.fill_diagonal(B, 0)
    return B


def erase_diagonal_blocks(A, row_range, col_range):
    """ remove diagonal blocks of the block matrix A """
    if len(row_range) != len(col_range):
        print('no. of column blocks != no. of row blocks!!')
    C = len(row_range) - 1
    B = A.copy()
    for c in range(C):
        B[row_range[c]: row_range[c+1], col_range[c]: col_range[c+1]] = 0
    return B


def inv_IpXY(X, Y):
    """
    Calculate the inverse of matrix A = I + XY.
    if X is a fat matrix (number of columns >> number of rows), then use inv(I + X*Y)
    else: use equation: (I + XY)^(-1) = I - Y*(I + Y*X)^(-1)*X
    """
    d1 = X.shape[0]
    d2 = X.shape[1]
    if d1 > d2:
        M = np.eye(d1) - np.dot(np.dot(X, LA.inv(np.eye(d2) + np.dot(Y, X))), Y)
    else:
        M = LA.inv(np.eye(d1) + np.dot(X, Y))
    return M


def progress_str(cur_val, max_val, total_point=50):
    p = int(math.ceil(float(cur_val)*total_point / max_val))
    return '|' + p*'#' + (total_point - p)*'.' + '|'


def get_time_str():
    print('Time now: ' + strftime("%m/%d/%Y %H:%M:%S"))
    return strftime("%m%d_%H%M%S")
