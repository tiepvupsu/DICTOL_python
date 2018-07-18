from __future__ import print_function
import numpy as np
import utils
from numpy import linalg as LA
import math

class Fista(object):
    def __init__(self):
        pass

    def solve2(self, Xinit = None, iterations = 100, tol = 1e-8, verbose = False):
        if Xinit is None:
            Xinit = np.zeros((self.D.shape[1], self.Y.shape[1]))
        Linv = 1/self.L
        lambdaLiv = self.lamb/self.L
        x_old = Xinit.copy()
        y_old = Xinit.copy()
        t_old = 1
        it = 0
        # cost_old = float("inf")
        for it in range(iterations):
            x_new = np.real(utils.shrinkage(y_old - Linv*self._grad(y_old), lambdaLiv))
            t_new = .5*(1 + math.sqrt(1 + 4*t_old**2))
            y_new = x_new + (t_old - 1)/t_new * (x_new - x_old)
            e = utils.norm1(x_new - x_old)/x_new.size
            if e < tol:
                break
            x_old = x_new.copy()
            t_old = t_new
            y_old = y_new.copy()
            if verbose:
                print('iter \t%d/%d, loss \t %4.4f'%(it + 1, iterations, self.lossF(x_new)))
        return x_new
        # self.DtY =

    def solve(self, Y, Xinit = None, iterations = 100, tol = 1e-8, verbose = False):
        self.fit(Y)
        if Xinit is None:
            Xinit = np.zeros((self.D.shape[1], self.Y.shape[1]))
        Linv = 1/self.L
        lambdaLiv = self.lamb/self.L
        x_old = Xinit.copy()
        y_old = Xinit.copy()
        t_old = 1
        it = 0
        # cost_old = float("inf")
        for it in range(iterations):
            x_new = np.real(utils.shrinkage(y_old - Linv*self._grad(y_old), lambdaLiv))
            t_new = .5*(1 + math.sqrt(1 + 4*t_old**2))
            y_new = x_new + (t_old - 1)/t_new * (x_new - x_old)
            e = utils.norm1(x_new - x_old)/x_new.size
            if e < tol:
                break
            x_old = x_new.copy()
            t_old = t_new
            y_old = y_new.copy()
            if verbose:
                print('iter \t%d/%d, loss \t %4.4f'%(it + 1, iterations, self.lossF(x_new)))
        return x_new


class Lasso(Fista):
    """
    Solving a Lasso problem using FISTA
    `X, = arg min_X 0.5*||Y - DX||_F^2 + lambd||X||_1
        = argmin_X f(X) + lambd||X||_1
        F(x) = f(X) + lamb||X||_1
    """
    def __init__(self, D, lamb = .1):
        self.D = D
        self.lamb = lamb
        self.DtD = np.dot(self.D.T, self.D)
        self.Y = None
        self.DtY = None
        self.L = np.max(LA.eig(self.DtD)[0])

    def fit(self, Y):
        self.Y = Y
        self.DtY = np.dot(self.D.T, self.Y)

    def _grad(self, X):
        return np.dot(self.DtD, X) - self.DtY

    def _calc_f(self, X):
        return 0.5*utils.normF2(self.Y - np.dot(self.D, X))

    def lossF(self, X):
        return self._calc_f(X) + self.lamb*utils.norm1(X)

def _test_lasso():
    d = 3
    N = 7
    k = 7
    Y = utils.normc(np.random.rand(d, N))
    Y1 = utils.normc(np.random.rand(d, N))
    D = utils.normc(np.random.rand(d, k))
    l = Lasso(D, lamb = .01)
    l.fit(Y)
    X = l.solve(verbose = True)
    print(X)
    X = Lasso(D).solve2(Y1, verbose = True)
    X = Lasso(D).solve2(Y1, Xinit = X, verbose = True)
    print(X)

   # X = l.solve()
if __name__ == '__main__':
    _test_lasso()
