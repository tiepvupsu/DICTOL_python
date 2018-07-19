from __future__ import print_function
import utils
import sparse_coding
import numpy as np
from ODL import ODL, ODL_updateD

class COPAR(object):
    def __init__(self, lambd = 0.01, eta = 0.0001, updateX_iters = 100, updateD_iters = 100):
        self.lambd = lambd
        self.eta = eta
        self.D = None
        self.X = None
        self.Y = None
        self.updateX_iters = updateX_iters
        self.updateD_iters = updateD_iters
        pass

    def _getYc(self, c):
        return utils.get_block_col(self.Y, c, self.Y_range)

    def _getDc(self, c):
        return utils.get_block_col(self.D, c, self.D_range_ext)

    def loss(self):
        """
        cost = COPAR_cost(Y, Y_range, D, D_range_ext, X, opts):
        Calculating cost function of COPAR with parameters lambda and eta are
        stored in `opts.lambda` and `opts.rho`.
        `f(D, X) = 0.5*sum_{c=1}^C 05*||Y - DX||_F^2 +
                      sum_{c=1}^C ( ||Y_c - D_Cp1 X^Cp1_c - D_c X_c^c||F^2 +
                  sum_{i != c}||X^i_c||_F^2) + lambda*||X||_1 +
                  0.5*eta*sum_{i \neq c}||Di^T*Dc||_F^2`
        -----------------------------------------------
        Author: Tiep Vu, thv102@psu.edu, 5/11/2016
                (http://www.personal.psu.edu/thv102/)
        -----------------------------------------------
        """
        cost = self.lambd*utils.norm1(self.X)
        cost1 = utils.normF2(self.Y - np.dot(self.D, self.X))
        DCp1 = self._getDc(self.nclass)
        # DCp1_range = range(self.D_range_ext[-2], self.D_range_ext[-1])
        for c in range(self.nclass):
            Dc = self._getDc(c)
            Yc = self._getYc(c)
            Xc = utils.get_block_col(self.X, c, self.Y_range)
            Xcc = utils.get_block_row(Xc, c, self.D_range_ext)
            XCp1c = utils.get_block_row(Xc, self.nclass, self.D_range_ext)

            cost1 += utils.normF2(Yc - np.dot(Dc, Xcc)) - np.dot(DCp1, XCp1c)
            XX = Xc[: self.D_range_ext[-2], :]
            XX = np.delete(XX, range(self.D_range_ext[c], self.D_range_ext[c+1]), axis = 0)
            cost1 += utils.normF2(XX)

        cost += cost1 + .5*self.eta*utils.normF2(\
                utils.erase_diagonal_blocks(np.dot(self.D.T, self.D), \
                self.D_range_ext, self.D_range_ext))
        return cost

    def fit(self, Y, label_train, k, k0, iterations = 100, verbose = False, show_after = 10):
        self.k = k
        self.k0 = k0
        self.Y = Y
        del Y
        self.Y_range = utils.label_to_range(label_train)
        self.nclass = self.Y_range.size - 1
        self.D_range = k*range(self.nclass+1)
        self.D_range_ext = self.D_range + [k*self.nclass + k0]
        # init
        self._initialize()
        for it in range(iterations):
            self._updateX()
            self._updateD()
        # pass

    def _initialize(self):
        for c in range(self.nclass):
            clf = ODL(lambd = self.lambd)
            clf.fit(self._getYc(c), self.k)
            self.D[:, self.D_range_ext[c]:self.D_range_ext[c+1]] = clf.D
            self.X[self.D_range_ext[c]:self.D_range_ext[c+1], \
                   self.Y_range[c]:self.Y_range[c+1]] = clf.X
        if self.k0 > 0:
            clf.fit(self.Y, self.k0)
            self.D[:, self.D_range_ext[-2]:self.D_range_ext[-1]] = clf.D
            self.X[self.D_range_ext[-2]:self.D_range_ext[-1]]

    def _updateX(self):
        pass

    def _updateD(self):
        Yhat = np.zeros_like(self.Y)
        for c in range(self.nclass):
            Yc = self._getYc(c)
            Dc = self._getDc(c)
            Xc = utils.get_block_col(self.X, c, self.Y_range)
            Xcc = utils.get_block_row(Xc, c, self.D_range_ext)
            XCp1c = utils.get_block_row(Xc, self.nclass, self.D_range_ext)
            Ychat = Yc - np.dot(self.D, Xc) + np.dot(Dc, Xcc)
            Ycbar = Yc - np.dot(DCp1, XCp1c)
            E = np.dot(Ychat + Ycbar, Xcc.T)
            F = 2*np.dot(Xcc, Xcc.T)
            A = D.copy()
            Dc_range = range(self.D_range_ext[c], self.D_range_ext[c+1])
            A = np.delete(A, Dc_range, axis = 1)
            self.D[:,Dc_range]
        pass

    def predict(self, Y):
        E = np.zeros((self.nclass, Y.shape[1]))
        for c in range(self.nclass):
            Dc = self._getDc(c)
            lasso = sparse_coding.Lasso(Dc, self.lambd)
            Xc = lasso.solve(Y)
            R1 = Y - np.dot(Dc, Xc)
            E[c, :] = 0.5*(R1*R1).sum(axis = 0) + self.lambd*abs(Xc).sum(axis = 0)
        return np.argmin(E, axis = 0) + 1

    def evaluate(self, Y_test, label_test, metrics = ['accuracy']):
        print('evaluating...')
        pred = self.predict(Y_test)
        acc = np.sum(pred == label_test)/float(utils.numel(label_test))
        print('accuracy = %.2f'%(100*acc))
        return acc


def _test_unit():
    dataset = 'myYaleB'
    N_train = 15
    dataset, Y_train, Y_test, label_train, label_test = \
           utils.train_test_split(dataset, N_train)
    clf = COPAR(lambd = 0.001, eta = 0.01)
    clf.fit(Y_train, label_train, k = 10, iterations = 100, verbose = True)
    clf.evaluate(Y_test, label_test)

if __name__ == '__main__':
    _test_unit()
