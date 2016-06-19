import numpy as np 
from numpy import linalg as LA 

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

def numel(A):
    """
    return number of elements of a numpy array 
    """
    return A.size 

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

def zeros(m, n):
    return np.zeros((m, n))

def ones(m, n):
    return np.ones((m, n))

def eye(n):
    return np.eye(n)