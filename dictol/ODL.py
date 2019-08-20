from __future__ import print_function
from . import utils, optimize
import numpy as np


class ODL(object):
    """
    Solving the optimization problem:
        (D, X) = arg min_{D, X} 0.5||Y - DX||_F^2 + lamb||X||_1
    """
    def __init__(self, k, lambd=0.001, updateD_iters=100, updateX_iters=100):
        self.lambd = lambd
        self.k = k
        self.Y = None
        self.D = None
        self.X = None
        self.updateD_iters = updateD_iters
        self.updateX_iters = updateX_iters

    def fit(self, Y, iterations=100, verbose=False):
        """
        Y: numpy data [n_features, n_samples]
        k: interger: number of atoms in the dictionary
            if k is None, select k = round(0.2*n_samples)
        """
        self.Y = Y
        Y_range = np.array([0, self.Y.shape[1]])
        D_range = np.array([0, self.k])
        self.D = utils.pickDfromY(self.Y, Y_range, D_range)
        self.X = np.zeros((self.D.shape[1], self.Y.shape[1]))
        for it in range(iterations):
            # update X
            lasso = optimize.Lasso(self.D, self.lambd)
            lasso.fit(self.Y, Xinit = self.X)
            self.X = lasso.coef_
            # update D
            F = np.dot(self.X, self.X.T)
            E = np.dot(self.Y, self.X.T)
            self.D = optimize.ODL_updateD(self.D, E, F, iterations = self.updateD_iters)
            if verbose:
                print('iter \t%d/%d \t\t loss \t%.4f'%(it, iterations, self.loss()))

    def loss(self):
        loss = 0.5*utils.normF2(self.Y - np.dot(self.D, self.X)) + \
                self.lambd*utils.norm1(self.X)
        return loss
