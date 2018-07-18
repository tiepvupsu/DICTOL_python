from __future__ import print_function
import utils
import sparse_coding
import numpy as np
# from numpy import linalg as LA
from ODL import ODL, ODL_updateD

class DLSI(object):
    def __init__(self, lambd = 0.01, eta = 0.01, updateD_iters = 100, updateX_iters = 100):
        self.lambd = 0.01
        self.eta = 0.01
        self.D = None
        self.X = None
        self.Y = None
        self.updateD_iters = updateD_iters
        self.updateX_iters = updateX_iters

    def fit(self, Y, label_train, k, iterations = 100, verbose = False):
        self.Y = Y
        del Y
        self.Y_range = utils.label_to_range(label_train)
        self.nclass = self.Y_range.size - 1
        self.D_range = k*np.arange(self.nclass + 1)
        self.D = np.zeros((self.Y.shape[0], self.D_range[-1]))
        self.X = []
        for c in range(self.nclass):
            n_rows = self.D_range[c+1] - self.D_range[c]
            n_cols = self.Y_range[c+1] - self.Y_range[c]
            self.X.append(np.zeros((n_rows, n_cols)))

        self._initialize()
        if verbose:
            print('initial loss = %.4f'%self.loss())

        for it in range(iterations):
            # update X
            self._updateX()
            # update D
            self._updateD()
            if verbose:
                print('iter \t %d/%d \t\t loss \t %.4f'%(it, iterations, self.loss()))

    def _getYc(self, c):
        return utils.get_block_col(self.Y, c, self.Y_range)

    def _getDc(self, c):
        return utils.get_block_col(self.D, c, self.D_range)

    def _initialize(self):
        for c in range(self.nclass):
            Yc = utils.get_block_col(self.Y, c, self.Y_range)
            clf = ODL(self.lambd)
            clf.fit(Yc, self.D_range[c+1] - self.D_range[c])
            # import pdb
            # pdb.set_trace()
            self.D[:, self.D_range[c]: self.D_range[c+1]] = clf.D
            self.X[c] = clf.X

    def _updateX(self):
        for c in range(self.nclass):
            lasso = sparse_coding.Lasso(self._getDc(c), self.lambd)
            # import pdb
            # pdb.set_trace()
            self.X[c] = lasso.solve(self._getYc(c), Xinit = self.X[c])

    def _updateD(self):
        for c in range(self.nclass):
            self._updateDc(c)

    def _updateDc(self, c):
        Dc = self._getDc(c)
        Yc = self._getYc(c)
        E = np.dot(Yc, self.X[c].T)
        F = np.dot(self.X[c], self.X[c].T)
        A = np.delete(self.D, range(self.D_range[c], self.D_range[c+1]), axis = 1).T
        """
        problem: `D = argmin_D -2trace(ED') + trace(FD'*D) + lambda *||A*D||F^2,`
        subject to: `||d_i||_2^2 <= 1`
        where F is a positive semidefinite matrix
        ========= aproach: ADMM ==============================
        rewrite: `[D, Z] = argmin -2trace(ED') + trace(FD'*D) + lambda ||A*Z||_F^2,`
            subject to `D = Z; ||d_i||_2^2 <= 1`
        aproach 1: ADMM.
        1. D = -2trace(ED') + trace(FD'*D) + rho/2 ||D - Z + U||_F^2,
            s.t. ||d_i||_2^2 <= 1
        2. Z = argmin lambda*||A*Z|| + rho/2||D - Z + U||_F^2
        3. U = U + D - Z
        solve 1: D = argmin -2trace(ED') + trace(FD'*D) + rho/2 ||D - W||_F^2
                              with W = Z - U;
                   = argmin -2trace((E - rho/2*W)*D') +
                      trace((F + rho/2 * eye())*D'D)
        solve 2: derivetaive: 0 = 2A'AZ + rho (Z - V) with V = D + U
        `Z = B*rhoV` with `B = (2*lambda*A'*A + rho I)^{-1}`
        `U = U + D - Z`
        -----------------------------------------------
        Author: Tiep Vu, thv102@psu.edu, 5/11/2016
                (http://www.personal.psu.edu/thv102/)
        -----------------------------------------------
        """
        rho = 1.0
        Z_old = Dc.copy()
        U = np.zeros_like(Dc)
        I_k = np.eye(Dc.shape[1])
        X = 2*self.lambd/rho*A.T
        Y = A.copy()
        B1 = np.dot(X, utils.inv_IpXY(Y, X))
        tol = 1e-8
        for it in range(self.updateD_iters):
            W = Z_old - U
            E2 = E + rho/2*W
            F2 = F + rho/2*I_k
            Dc = ODL_updateD(Dc, E2, F2)
            V = Dc + U
            Z_new = rho*(V - np.dot(B1, np.dot(Y, V)))
            e1 = utils.normF2(Dc - Z_new)
            e2 = rho*utils.normF2(Z_new - Z_old)
            if e1 < tol and e2 < tol:
                break
            U = U + Dc - Z_new
            Z_old = Z_new.copy()
        self.D[:, self.D_range[c]:self.D_range[c+1]] = Dc


    def loss(self):
        cost = 0
        for c in range(self.nclass):
            Yc = utils.get_block_col(self.Y, c, self.Y_range)
            Xc = self.X[c]
            Dc = utils.get_block_col(self.D, c, self.D_range)
            cost += 0.5*utils.normF2(Yc - np.dot(Dc, Xc)) + self.lambd*utils.norm1(Xc)
        cost += 05*self.eta*utils.normF2(\
                utils.erase_diagonal_blocks(np.dot(self.D.T, self.D), self.D_range, self.D_range))
        return cost

    def predict(self, Y_test):
        pass

    def evaluate(self, Y_test, label_test, metrics = ['accuracy']):
        pass


def _test_unit():
    dataset = 'myYaleB'
    N_train = 10
    dataset, Y_train, Y_test, label_train, label_test = \
           utils.train_test_split(dataset, N_train)
    clf = DLSI(lambd = 0.01)
    clf.fit(Y_train, label_train, k = 5, iterations = 100, verbose = True)
    # clf.evaluate(Y_test, label_test)



if __name__ == '__main__':
    _test_unit()
