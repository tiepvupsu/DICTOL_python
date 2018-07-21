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

