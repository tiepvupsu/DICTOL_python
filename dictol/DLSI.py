from __future__ import print_function
from . import utils, optimize, base
import numpy as np
from .ODL import ODL

class DLSI(base.BaseModel):
    def __init__(self, k=10, lambd=0.01, eta=0.01, updateD_iters=100, updateX_iters=100):
        self.lambd = 0.01
        self.eta = 0.01
        self.D = None
        self.X = None
        self.Y = None
        self.k = k
        self.updateD_iters = updateD_iters
        self.updateX_iters = updateX_iters

    def fit(self, Y, label_train, iterations=100, verbose=False, show_after=10):
        self.Y = Y
        self.Y_range = utils.label_to_range(label_train)
        self.num_classes = len(self.Y_range) - 1
        self.D_range = [self.k * i for i in range(self.num_classes + 1)]
        self.D = np.zeros((self.Y.shape[0], self.D_range[-1]))
        self.X = []
        for c in range(self.num_classes):
            n_rows = self.D_range[c+1] - self.D_range[c]
            n_cols = self.Y_range[c+1] - self.Y_range[c]
            self.X.append(np.zeros((n_rows, n_cols)))
        if verbose:
            print('Initializing ... ')

        self._initialize()
        if verbose:
            print('initial loss = %.4f' % self.loss())

        for it in range(iterations):
            # update X
            self._updateX()
            # update D
            self._updateD()
            if verbose and (it == 0 or (it+1) % show_after == 0):
                print('iter \t %3d/%d \t loss \t %.4f' % (it + 1, iterations, self.loss()))

    def _getYc(self, c):
        return utils.get_block_col(self.Y, c, self.Y_range)

    def _getDc(self, c):
        return utils.get_block_col(self.D, c, self.D_range)

    def _initialize(self):
        for c in range(self.num_classes):
            Yc = utils.get_block_col(self.Y, c, self.Y_range)
            clf = ODL(k=self.D_range[c+1] - self.D_range[c], lambd=self.lambd)
            clf.fit(Yc)
            self.D[:, self.D_range[c]: self.D_range[c+1]] = clf.D
            self.X[c] = clf.X

    def _updateXc(self, c):
        lasso = optimize.Lasso(self._getDc(c), self.lambd)
        lasso.fit(self._getYc(c), Xinit=self.X[c])
        self.X[c] = lasso.coef_

    def _updateX(self):
        for c in range(self.num_classes):
            self._updateXc(c)

    def _updateD(self):
        for c in range(self.num_classes):
            self._updateDc(c)

    def _updateDc(self, c):
        Dc = self._getDc(c)
        Yc = self._getYc(c)
        E = np.dot(Yc, self.X[c].T)
        F = np.dot(self.X[c], self.X[c].T)
        A = np.delete(self.D, list(range(self.D_range[c], self.D_range[c+1])), axis=1).T

        self.D[:, self.D_range[c]:self.D_range[c+1]] = optimize.DLSI_updateD(Dc, E, F, A, self.lambd)

    def loss(self):
        cost = 0
        for c in range(self.num_classes):
            Yc = utils.get_block_col(self.Y, c, self.Y_range)
            Xc = self.X[c]
            Dc = utils.get_block_col(self.D, c, self.D_range)
            cost += 0.5*utils.normF2(Yc - np.dot(Dc, Xc)) + self.lambd*utils.norm1(Xc)

        cost += 0.5*self.eta*utils.normF2(utils.erase_diagonal_blocks(
            np.dot(self.D.T, self.D),
            self.D_range,
            self.D_range
        ))
        return cost

    def predict(self, Y):
        E = np.zeros((self.num_classes, Y.shape[1]))
        for c in range(self.num_classes):
            Dc = self._getDc(c)
            lasso = optimize.Lasso(Dc, self.lambd)
            lasso.fit(Y)
            Xc = lasso.coef_
            R1 = Y - np.dot(Dc, Xc)
            E[c, :] = 0.5*(R1*R1).sum(axis=0) + self.lambd*abs(Xc).sum(axis=0)
        return np.argmin(E, axis=0) + 1


def mini_test_unit():
    """
    mini test on simulated data
    """
    print('\n================================================================')
    print('Mini Unit test: DLSI')
    dataset = 'myYaleB'
    N_train = 5
    Y_train, Y_test, label_train, label_test = utils.train_test_split(dataset, N_train)
    clf = DLSI(k=3, lambd=0.001, eta=0.001)
    clf.fit(Y_train, label_train, iterations=10, verbose=True)
    clf.evaluate(Y_test, label_test)


def test_unit():
    print('\n===================================================================')
    print('Unit test: DLSI')
    dataset = 'myYaleB'
    N_train = 15
    Y_train, Y_test, label_train, label_test = utils.train_test_split(dataset, N_train)
    clf = DLSI(k=10, lambd=0.001, eta=0.001)
    clf.fit(Y_train, label_train, iterations=100, verbose=True)
    clf.evaluate(Y_test, label_test)


if __name__ == '__main__':
    mini_test_unit()
