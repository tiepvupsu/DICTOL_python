from __future__ import print_function
from . import utils, optimize, base
import numpy as np
from .ODL import ODL


class UpdateXc(optimize.Fista):
    """
    Update Xc in COPAR (page 189-190 COPAR)
    see COPAR paper:
    http://www.cs.zju.edu.cn/people/wangdh/papers/draft_ECCV12_particularity.pdf

     cost = normF2(Yc - D*Xc) + normF2(Yc - DcXcc - DCp1*XCp1c) +
              sum_{i \neq c, 1 \leq i \leq C} normF2(Xic);
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 5/12/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """
    def __init__(self, D, D_range_ext, Y, Y_range, lambd, iterations = 100):
        self.D = D
        self.lambd = lambd
        self.DtD = np.dot(self.D.T, self.D)
        self.Y = Y
        self.Y_range = Y_range
        self.nclass = len(D_range_ext) - 2
        self.DtY = np.dot(D.T, Y)
        self.DCp1 = utils.get_block_col(D, self.nclass, D_range_ext)
        self.DCp1tDCp1 = np.dot(self.DCp1.T, self.DCp1)
        self.D_range_ext = D_range_ext
        self.k0 = D_range_ext[-1] - D_range_ext[-2]
        if self.k0 > 0:
            self.L = utils.max_eig(self.DtD) + utils.max_eig(self.DCp1tDCp1)
        else:
            self.L = utils.max_eig(self.DtD)
        self.c = -1
        self.DCp1 = utils.get_block_col(D, self.nclass, self.D_range_ext)

    def set_class(self, c):
        self.c = c
        self.Yc = utils.get_block_col(self.Y, c, self.Y_range)
        self.Dc = utils.get_block_col(self.D, c, self.D_range_ext)
        self.DctDc = utils.get_block(self.DtD, c, c, self.D_range_ext, self.D_range_ext)
        self.DCp1tDc = utils.get_block(
            self.DtD, self.nclass, c, self.D_range_ext, self.D_range_ext)
        self.DtYc = utils.get_block_col(self.DtY, c, self.Y_range)
        self.DtYc2 = self.DtYc.copy()

        self.DtYc2[self.D_range_ext[c]:self.D_range_ext[c+1], :] = \
            2*self.DtYc[self.D_range_ext[c]: self.D_range_ext[c+1], :]

        self.DtYc2[self.D_range_ext[-2]:self.D_range_ext[-1], :] = \
            2*self.DtYc[self.D_range_ext[-2]:self.D_range_ext[-1], :]

    def _grad(self, Xc0):
        Xc = Xc0.copy()
        c = self.c
        g0 = np.dot(self.DtD, Xc)
        Xcc = utils.get_block_row(Xc, self.c, self.D_range_ext)
        XCp1c = utils.get_block_row(Xc, self.nclass, self.D_range_ext)
        if self.k0 > 0:
            Xc[self.D_range_ext[c]: self.D_range_ext[c+1], :] = \
                np.dot(self.DctDc, Xcc) + np.dot(self.DCp1tDc.T, XCp1c)
            Xc[self.D_range_ext[-2]: self.D_range_ext[-1], :] = \
                np.dot(self.DCp1tDCp1, XCp1c) + np.dot(self.DCp1tDc, Xcc)
        else:
            Xc[self.D_range_ext[c]: self.D_range_ext[c+1], :] = np.dot(self.DctDc, Xcc)
        return g0 + Xc - self.DtYc2

    def _calc_f(self, Xc):
        """
        optimize later
        """
        Xcc = utils.get_block_row(Xc, self.c, self.D_range_ext)
        XCp1c = utils.get_block_row(Xc, self.nclass, self.D_range_ext)
        cost = utils.normF2(self.Yc - np.dot(self.D, Xc))
        cost += utils.normF2(self.Yc - np.dot(self.Dc, Xcc) - np.dot(self.DCp1, XCp1c))
        for i in range(self.nclass):
            if i != self.c:
                Xic = utils.get_block_row(Xc, i, self.D_range_ext)
                cost += utils.normF2(Xic)
        return .5*cost

    def lossF(self, Xc):
        return self._calc_f(Xc) + utils.norm1(Xc)


class COPAR(base.BaseModel):
    def __init__(self, k, k0, lambd = 0.01, eta = 0.0001, updateX_iters = 100, updateD_iters = 100):
        self.k = k
        self.k0 = k0
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
        for c in range(self.nclass):
            Dc = self._getDc(c)
            Yc = self._getYc(c)
            Xc = utils.get_block_col(self.X, c, self.Y_range)
            Xcc = utils.get_block_row(Xc, c, self.D_range_ext)
            XCp1c = utils.get_block_row(Xc, self.nclass, self.D_range_ext)

            cost1 += utils.normF2(Yc - np.dot(Dc, Xcc) - np.dot(DCp1, XCp1c))
            XX = Xc[: self.D_range_ext[-2], :]
            XX = np.delete(XX, list(range(self.D_range_ext[c], self.D_range_ext[c+1])), axis=0)
            cost1 += utils.normF2(XX)

        cost += cost1 + .5*self.eta*utils.normF2(
            utils.erase_diagonal_blocks(np.dot(self.D.T, self.D),
                                        self.D_range_ext,
                                        self.D_range_ext))
        return cost

    def fit(self, Y, label_train, iterations=100, verbose=False, show_after=5):
        self.Y = Y
        del Y
        self.Y_range = utils.label_to_range(label_train)
        self.nclass = len(self.Y_range) - 1
        D_range = [self.k*i for i in range(self.nclass+1)]
        self.D_range_ext = D_range + [self.k*self.nclass + self.k0]
        # init
        if verbose:
            print('initializing ... ')
        self._initialize()
        if verbose:
            print('initialization cost = %.4f'%self.loss())
        for it in range(iterations):
            self._updateD()
            self._updateX()
            if verbose and (it == 0 or (it + 1) % show_after == 0):
                print('iter \t%3d/%3d \t loss %.4f'%(it+1, iterations, self.loss()))

    def _initialize(self):
        self.D = np.zeros((self.Y.shape[0], self.D_range_ext[-1]))
        self.X = np.zeros((self.D_range_ext[-1], self.Y.shape[1]))
        for c in range(self.nclass):
            clf = ODL(k=self.k, lambd=self.lambd)
            clf.fit(self._getYc(c))
            self.D[:, self.D_range_ext[c]:self.D_range_ext[c+1]] = clf.D
            self.X[self.D_range_ext[c]:self.D_range_ext[c+1], \
                   self.Y_range[c]:self.Y_range[c+1]] = clf.X
        if self.k0 > 0:
            clf = ODL(k=self.k0, lambd=self.lambd)
            clf.fit(self.Y)
            self.D[:, self.D_range_ext[-2]:self.D_range_ext[-1]] = clf.D
            self.X[self.D_range_ext[-2]:self.D_range_ext[-1]]

    def _updateX(self):
        updatxc = UpdateXc(
            self.D, self.D_range_ext, self.Y, self.Y_range, self.lambd, iterations=100)
        for c in range(self.nclass):
            updatxc.set_class(c)
            Xc = utils.get_block_col(self.X, c, self.Y_range)
            # updatxc.check_grad(Xc)
            self.X[:, self.Y_range[c]: self.Y_range[c+1]] = updatxc.solve(Xinit=Xc)

    def _updateD(self):
        Yhat = np.zeros_like(self.Y)
        DCp1 = self._getDc(self.nclass)
        for c in range(self.nclass):
            Dc_range = list(range(self.D_range_ext[c], self.D_range_ext[c+1]))
            Yc_range = list(range(self.Y_range[c], self.Y_range[c+1]))
            Yc = self._getYc(c)
            Dc = self._getDc(c)
            Xc = utils.get_block_col(self.X, c, self.Y_range)
            Xcc = utils.get_block_row(Xc, c, self.D_range_ext)
            XCp1c = utils.get_block_row(Xc, self.nclass, self.D_range_ext)
            Ychat = Yc - np.dot(self.D, Xc) + np.dot(Dc, Xcc)
            Ycbar = Yc - np.dot(DCp1, XCp1c)
            E = np.dot(Ychat + Ycbar, Xcc.T)
            F = 2*np.dot(Xcc, Xcc.T)
            A = self.D.copy()
            A = np.delete(A, Dc_range, axis=1)
            self.D[:,Dc_range] = optimize.DLSI_updateD(Dc, E, F, A.T, self.eta)
            Yhat[:, Yc_range] = Yc - np.dot(self.D[:, Dc_range], Xcc)
        ## DCp1
        XCp1 = utils.get_block_row(self.X, self.nclass, self.D_range_ext)
        Ybar = self.Y - np.dot(self.D[:, : self.D_range_ext[-2]],
                               self.X[: self.D_range_ext[-2], :])
        E = np.dot(Ybar + Yhat, XCp1.T)
        F = 2*np.dot(XCp1, XCp1.T)
        A = self.D[:, : self.D_range_ext[-2]]
        DCp1_range = list(range(self.D_range_ext[-2], self.D_range_ext[-1]))
        self.D[:, DCp1_range] = optimize.DLSI_updateD(self.D[:, DCp1_range], E, F, A.T, self.eta)

    def predict(self, Y):
        E = np.zeros((self.nclass, Y.shape[1]))
        for c in range(self.nclass):
            Dc = self._getDc(c)
            lasso = optimize.Lasso(Dc, self.lambd)
            lasso.fit(Y)
            Xc = lasso.solve()
            R1 = Y - np.dot(Dc, Xc)
            E[c, :] = 0.5*(R1*R1).sum(axis=0) + self.lambd*abs(Xc).sum(axis=0)
        return np.argmin(E, axis=0) + 1


def mini_test_unit():
    print('\n================================================================')
    print('Mini Unit test: COPAR')
    dataset = 'myYaleB'
    N_train = 5
    Y_train, Y_test, label_train, label_test = utils.train_test_split(dataset, N_train)
    clf = COPAR(k=4, k0=5, lambd=0.001, eta=0.01)
    clf.fit(Y_train, label_train, iterations=10, verbose=True)
    clf.evaluate(Y_test, label_test)


def test_unit():
    print('\n================================================================')
    print('Mini Unit test: COPAR')
    dataset = 'myYaleB'
    N_train = 15
    Y_train, Y_test, label_train, label_test = utils.train_test_split(dataset, N_train)
    clf = COPAR(k=10, k0=5, lambd=0.001, eta=0.01)
    clf.fit(Y_train, label_train, iterations=100, verbose=True)
    clf.evaluate(Y_test, label_test)


if __name__ == '__main__':
    mini_test_unit()
