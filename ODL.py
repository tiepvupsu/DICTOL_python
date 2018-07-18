from __future__ import print_function
import utils
import sparse_coding
import numpy as np
from numpy import linalg as LA

def ODL_updateD(D, E, F, iterations = 100, tol = 1e-8):
    """
    * The main algorithm in ODL.
    * Solving the optimization problem:
      `D = arg min_D -2trace(E'*D) + trace(D*F*D')` subject to: `||d_i||_2 <= 1`,
         where `F` is a positive semidefinite matrix.
    * Syntax `[D, iter] = ODL_updateD(D, E, F, opts)`
      - INPUT:
        + `D, E, F` as in the above problem.
        + `opts`. options:
          * `iterations`: maximum number of iterations.
          * `tol`: when the difference between `D` in two successive
                    iterations less than this value, the algorithm will stop.
      - OUTPUT:
        + `D`: solution.
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/07/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """
    def calc_cost(D):
        return -2*np.trace(np.dot(E, D.T)) + np.trace(np.dot(np.dot(F, D.T), D))

    D_old = D.copy()
    it = 0
    sizeD = utils.numel(D)
    # print opts.tol
    # while it < opts.max_iter:
    for it in range(iterations):
        for i in xrange(D.shape[1]):
            if F[i,i] != 0:
                a = 1.0/F[i,i] * (E[:, i] - D.dot(F[:, i])) + D[:, i]
                D[:,i] = a/max(LA.norm(a, 2), 1)

        if LA.norm(D - D_old, 'fro')/sizeD < tol:
            break
        D_old = D.copy()
    return D


class ODL(object):
    """
    Solving the optimization problem:
        (D, X) = arg min_{D, X} 0.5||Y - DX||_F^2 + lamb||X||_1
    """
    def __init__(self, lamb = 0.001, updateD_iters = 100, updateX_iters = 100):
        self.lamb = lamb
        self.Y = None
        self.D = None
        self.X = None
        self.updateD_iters = updateD_iters
        self.updateX_iters = updateX_iters

    def fit(self, Y, k = None, iterations = 100, verbose = False):
        """
        Y: numpy data [n_features, n_samples]
        k: interger: number of atoms in the dictionary
            if k is None, select k = round(0.2*n_samples)
        """
        if k is None:
            k = int(0.2*Y.shape[1])
        self.Y = Y
        del Y
        Y_range = np.array([0, self.Y.shape[1]])
        D_range = np.array([0, k])
        self.D = utils.pickDfromY(self.Y, Y_range, D_range)
        self.X = np.zeros((self.D.shape[1], self.Y.shape[1]))
        for it in range(iterations):
            # update X
            self.X = sparse_coding.Lasso(self.D, self.lamb).solve(self.Y, \
                    iterations = self.updateX_iters)
            # update D
            F = np.dot(self.X, self.X.T)
            E = np.dot(self.Y, self.X.T)
            self.D = ODL_updateD(self.D, E, F, iterations = self.updateD_iters)
            if verbose:
                print('iter \t%d/%d \t\t loss \t%.4f'%(it, iterations, self.loss()))

    def loss(self):
        l = 0.5*utils.normF2(self.Y - np.dot(self.D, self.X)) + \
                self.lamb*utils.norm1(self.X)
        return l


def _test_unit():
    d = 100
    N = 500
    k = 200
    Y = utils.normc(np.random.randn(d, N))
    clf = ODL(lamb = 0.01)
    clf.fit(Y, k, verbose = True, iterations = 300)
    # print(clf.X)

if __name__ == '__main__':
    _test_unit()
