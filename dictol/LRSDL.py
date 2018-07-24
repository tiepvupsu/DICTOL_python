"""
Low-rank Shared Dictionary Learning
and Fisher Discriminant Dictionary Learning
"""
from __future__ import print_function
import optimize, matlab_syntax
import utils
from utils import normF2, norm1, get_block_col, get_block_row, nuclearnorm, build_mean_matrix
import numpy as np
from ODL import ODL

_zero = np.array([0])

# supporting function
class _UpdateXX0(optimize.Fista):
    """
    solve XX0 in LRSDL using Fista
    """
    def __init__(self, Y, Y_range, D, D_range, D0, k0, lambd = 0.01, lambd2 = 0.01):
        self.Y = Y
        self.Y_range = Y_range
        self.nclass = len(Y_range) - 1
        self.D = D
        self.D_range = D_range
        self.D0 = D0
        self.k0 = k0
        self.lambd = lambd
        self.lambd2 = lambd2
        self.DtD = np.dot(D.T, D)
        self.D_0 = utils.buildMhat(self.DtD, self.D_range, self.D_range)
        self.Dhat = self.D_0 + 2*self.lambd2*np.eye(self.D_0.shape[1])
        self.D0tD0 = np.dot(self.D0.T, self.D0) if self.k0 > 0 else _zero
        self.A = 2*self.D0tD0 + self.lambd2*np.eye(self.k0) if self.k0 > 0 else _zero
        self.DtD0 = np.dot(D.T, D0) if self.k0 > 0 else _zero
        self.D0tD = self.DtD0.T
        self.D0tY2 = 2*np.dot(self.D0.T, self.Y) if self.k0 > 0 else _zero

        self.DtY0 = np.dot(self.D.T, self.Y)
        self.L = utils.max_eig(self.Dhat) + 4*self.lambd2 + 1
        if self.k0 > 0: self.L += utils.max_eig(self.A)
        pass

    def fit(self):
        pass

    def _extract_fromX1(self, X1):
        K = self.D_range[-1]
        return (X1[:K, :], X1[K:, :]) if self.k0 > 0 else (X1[:K, :], _zero)

    def _grad(self, X1):
        X, X0 = self._extract_fromX1(X1)
        DtY = self.DtY0 - np.dot(self.DtD0, X0)
        Y_0 = utils.buildMhat(DtY, self.D_range, self.Y_range)
        g = np.dot(self.Dhat, X) - Y_0 + utils.buildM_2Mbar(X, self.Y_range, self.lambd2)
        if self.k0 > 0:
            g0 = np.dot(self.A, X0) - self.D0tY2 \
                + np.dot(self.D0tD, utils.buildMhat(X, self.D_range, self.Y_range)) \
                - self.lambd2*utils.buildMean(X0)
            return np.vstack((g, g0))
        return g

    def _fidelity(self, X):
        """
        * Calculating the fidelity term in FDDL[[4]](#fn_fdd):
        * $\sum_{c=1}^C \Big(\|Y_c - D_cX^c_c\|_F^2 +
            \sum_{i \neq c} \|D_c X^c_i\|_F^2\Big)$
        """
        cost = 0
        Y = self.Y
        for c in xrange(self.nclass):
            Yc   = get_block_col(Y, c, self.Y_range)
            Dc   = get_block_col(self.D, c, self.D_range)
            Xc   = get_block_row(X, c, self.D_range)
            Xcc  = get_block_col(Xc, c, self.Y_range)
            cost +=normF2(Yc - np.dot(Dc, Xcc))
            for i in xrange(self.nclass):
                if i == c:
                    continue
                Xci = get_block_col(Xc, i, self.Y_range)
                cost += normF2(np.dot(Dc, Xci))
        return cost

    def _discriminative(self, X):
        """
        * calculating the discriminative term in
        * $\|X\|_F^2 + \sum_{c=1}^C (\|Xc - Mc\|_F^2 - \|Mc - M\|_F^2) $
        """
        cost = normF2(X)
        m = np.mean(X, axis = 1)
        for c in xrange(self.nclass):
            Xc   = get_block_col(X, c, self.Y_range)
            Mc   = build_mean_matrix(Xc)
            cost += normF2(Xc - Mc)
            M    = matlab_syntax.repmat(m, 1, Xc.shape[1])
            cost -= normF2(Mc - M)
        return cost

    def _calc_f(self, X1):
        X, X0 = self._extract_fromX1(X1)
        Ybar = self.Y - np.dot(self.D0, X0)
        cost = 0.5*(normF2(Ybar - np.dot(self.D, X)) + self._fidelity(X)) + \
                0.5*self.lambd2*self._discriminative(X) + normF2(X0 - utils.buildMean(X0))
        return cost

    def lossF(self, X1):
        return self._calc_f(X1) + self.lambd*norm1(X1)

class LRSDL(object):
    def __init__(self, lambd = 0.01, lambd2 = 0.01, eta = 0.0001,
            k = 10, k0 = 5, updateX_iters = 100, updateD_iters = 100):
        self.lambd = lambd
        self.lambd2 = lambd2
        self.eta = eta
        self.D = None
        self.X = None
        self.Y = None
        self.k = k
        self.k0 = k0
        self.updateX_iters = updateX_iters
        self.updateD_iters = updateD_iters
        self.D_range = None
        self.D0 = None
        self.Y_range = None
        self.X = None
        self.X0 = None

    def _getYc(self, c):
        return get_block_col(self.Y, c, self.Y_range)

    def fit(self, Y, train_label, verbose = False, iterations = 100, show_after = 5):
        self.Y_range = utils.label_to_range(train_label)
        self.nclass = len(self.Y_range) - 1
        self.D_range = [self.k*i for i in range(self.nclass+1)]
        self.Y = Y
        self.D = np.zeros((self.Y.shape[0], self.D_range[-1]))
        self.X = np.zeros((self.D_range[-1], self.Y.shape[1]))
        if self.k0 > 0:
            self.D0 = np.zeros((self.Y.shape[0], self.k0))
            self.X0 = np.zeros((self.k0, self.Y.shape[1]))
        ## init
        if verbose:
            print('initializing ... ')
        self._initialize()
        if verbose:
            print('initial loss %.4f'%self.loss())
        ## train
        for it in range(iterations):
            # update D
            self._updateD()
            # update D0
            if self.k0 > 0:
               self._updateD0()
            self._updateXX0()
            if verbose and (it == 0 or (it + 1)%show_after == 0):
                print('iter \t%3d/%3d \t loss %.4f'%(it+1, iterations, self.loss()))

    def _updateD(self):
        Y = self.Y if self.k0 == 0 else self.Y - np.dot(self.D0, self.X0)
        F = utils.buildMhat(np.dot(self.X, self.X.T), self.D_range, self.D_range)
        E = np.dot(Y, utils.buildMhat(self.X.T, self.Y_range, self.D_range))
        self.D = optimize.ODL_updateD(self.D, E, F)

    def _extract_fromX1(self, X1):
        K = self.D_range[-1]
        return (X1[:K, :], X1[K:, :]) if self.k0 > 0 else (X1[:K, :], _zero)

    def _updateXX0(self):
        clf = _UpdateXX0(self.Y, self.Y_range, self.D, self.D_range, self.D0, \
                self.k0, lambd = self.lambd, lambd2 = self.lambd2)
        # clf.check_grad(np.vstack((self.X, self.X0)) if self.k0 > 0 else self.X)
        X1 = np.vstack((self.X, self.X0)) if self.k0 > 0 else self.X
        X1 = clf.solve(Xinit = X1)
        self.X, self.X0 = self._extract_fromX1(X1)

    def _buildYhat(self):
        """
        Yhat = [Yhat_1, Yhat_2, ..., Yhat_C]
        where Yhat_c = Yc - Dc*Xcc
        """
        Yhat = np.zeros_like(self.Y)
        for c in xrange(self.nclass):
            Yc = get_block_col(self.Y, c, self.Y_range)
            Dc = get_block_col(self.D, c, self.D_range)
            Xcc = utils.get_block(self.X, c, c, self.D_range, self.Y_range)
            Yhat[:, self.Y_range[c]: self.Y_range[c+1]] = Yc - np.dot(Dc, Xcc)
        return Yhat

    def _updateD0(self):
        Ybar = self.Y - np.dot(self.D, self.X)
        L = (Ybar + self._buildYhat())/2
        self.D0 = optimize.min_rank_dict(L, self.X0, self.eta/2, \
                Dinit = self.D0, iterations = 50)

    def _initialize(self):
        for c in range(self.nclass):
            clf = ODL(lambd = self.lambd, k = self.D_range[c+1] - self.D_range[c])
            clf.fit(self._getYc(c), iterations = 5)
            self.D[:, self.D_range[c]:self.D_range[c+1]] = clf.D
            self.X[self.D_range[c]: self.D_range[c+1], \
                   self.Y_range[c]: self.Y_range[c+1]] = clf.X

        if self.k0 > 0:
            odl = ODL(lambd = self.lambd, k = self.k0)
            odl.fit(self.Y)
            self.D0 = odl.D
            self.X0 = odl.X

    def _fidelity(self):
        """
        * Calculating the fidelity term in FDDL
        * $\sum_{c=1}^C \Big(\|Y_c - D_cX^c_c\|_F^2 +
            \sum_{i \neq c} \|D_c X^c_i\|_F^2\Big)$
        """
        cost = 0
        Y = self.Y - np.dot(self.D0, self.X0) if self.k0 > 0 else self.Y.copy()
        for c in xrange(self.nclass):
            Yc   = get_block_col(Y, c, self.Y_range)
            Dc   = get_block_col(self.D, c, self.D_range)
            Xc   = get_block_row(self.X, c, self.D_range)
            Xcc  = get_block_col(Xc, c, self.Y_range)
            cost +=normF2(Yc - np.dot(Dc, Xcc))
            for i in xrange(self.nclass):
                if i == c:
                    continue
                Xci = get_block_col(Xc, i, self.Y_range)
                cost += normF2(np.dot(Dc, Xci))
        return cost

    def _coefMM0(self):
        X1 = self.X if self.k0 == 0 else np.vstack((self.X, self.X0))
        return utils.build_mean_vector(X1, self.Y_range)

    def _discriminative(self):
        """
        * calculating the discriminative term in FDDL[[4]](#fn_fdd):
        * $\|X\|_F^2 + \sum_{c=1}^C (\|Xc - Mc\|_F^2 - \|Mc - M\|_F^2) $
        """
        cost = normF2(self.X)
        m = np.mean(self.X, axis = 1)
        for c in xrange(self.nclass):
            Xc   = get_block_col(self.X, c, self.Y_range)
            Mc   = build_mean_matrix(Xc)
            cost += normF2(Xc - Mc)
            M    = matlab_syntax.repmat(m, 1, Xc.shape[1])
            cost -= normF2(Mc - M)
        return cost

    def loss(self):
        Y = self.Y.copy()
        if self.k0 > 0:
            Y -= np.dot(self.D0, self.X0)
        cost = 0.5*normF2(Y - np.dot(self.D, self.X)) + \
                0.5*self._fidelity() + \
                0.5*self.lambd2*self._discriminative() + \
                self.lambd*norm1(self.X)

        if self.k0 > 0:
            cost += self.lambd*norm1(self.X0) + \
                    0.5*self.lambd2*normF2(self.X0 - build_mean_matrix(self.X0)) \
                    + self.eta*nuclearnorm(self.D0)
        return cost

    def predict(self, Y):
        N = Y.shape[1]
        lambda_list = [self.lambd]
        for lambd in lambda_list:
            E = np.zeros((self.nclass, N))
            for c in range(self.nclass):
                # Dc in D only
                Dc_ = get_block_col(self.D, c, self.D_range)
                # Dc in D and D0
                Dc = np.hstack((Dc_, self.D0)) if self.k0 > 0 else Dc_
                lasso = optimize.Lasso(Dc, lambd = lambd)
                lasso.fit(Y)
                Xc = lasso.solve()
                R = Y - np.dot(Dc, Xc)
                E[c, :] = 0.5*np.sum(R*R, axis = 0) + \
                        lambd*np.sum(np.abs(Xc), axis = 0)
            pred = np.argmin(E, axis = 0) + 1
        return pred
        pass

    def evaluate(self, Y_test, label_test):
        print('evaluating...')
        pred = self.predict(Y_test)
        acc = np.sum(pred == label_test)/float(label_test.size)
        print('accuracy = %.2f'%(100*acc))
        return acc


def mini_test_unit():
    print('\n===================================================================')
    print('Mini Unit test: Low-rank shared Dictionary Learning')
    dataset = 'myYaleB'
    N_train = 5
    dataset, Y_train, Y_test, label_train, label_test = \
           utils.train_test_split(dataset, N_train)
    clf = LRSDL(lambd = 0.01, lambd2 = 0.01, eta = 0.1, k = 4, k0 = 5)
    clf.fit(Y_train, label_train, iterations = 30, verbose = True)
    clf.evaluate(Y_test, label_test)

def mini_test_unit_FDDL():
    print('\n===================================================================')
    print('Mini Unit test: Fisher Disrciminant Dicationary Learning')
    dataset = 'myYaleB'
    N_train = 5
    dataset, Y_train, Y_test, label_train, label_test = \
           utils.train_test_split(dataset, N_train)
    clf = LRSDL(lambd = 0.01, lambd2 = 0.01, eta = 0.1, k = 4, k0 = 0)
    clf.fit(Y_train, label_train, iterations = 30, verbose = True)
    clf.evaluate(Y_test, label_test)

def test_unit_FDDL():
    print('\n===================================================================')
    print('Unit test: Fisher Disrciminant Dicationary Learning')
    dataset = 'myYaleB'
    N_train = 30
    dataset, Y_train, Y_test, label_train, label_test = \
           utils.train_test_split(dataset, N_train)
    clf = LRSDL(lambd = 0.01, lambd2 = 0.01, eta = 0.1, k = 20, k0 = 0)
    clf.fit(Y_train, label_train, iterations = 30, verbose = True)
    clf.evaluate(Y_test, label_test)

def test_unit():
    print('\n===================================================================')
    print('Unit test: Low-rank shared Dictionary Learning')
    dataset = 'myYaleB'
    N_train = 30
    dataset, Y_train, Y_test, label_train, label_test = \
           utils.train_test_split(dataset, N_train)
    clf = LRSDL(lambd = 0.01, lambd2 = 0.01, eta = 0.1, k = 20, k0 = 10)
    clf.fit(Y_train, label_train, iterations = 30, verbose = True)
    clf.evaluate(Y_test, label_test)

if __name__ == '__main__':
    mini_test_unit_FDDL()
    mini_test_unit()
    test_unit_FDDL()
    test_unit()

