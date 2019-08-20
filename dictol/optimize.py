from __future__ import print_function
import numpy as np
from . import utils
from numpy import linalg as LA
import math

def ODL_updateD(D, E, F, iterations=100, tol=1e-8):
    """
    The main algorithm in ODL.
    Solving the optimization problem:
      D = arg min_D -2trace(E'*D) + trace(D*F*D') subject to: ||d_i||_2 <= 1,
         where F is a positive semidefinite matrix.

    Parameters:
    -----------
    D, E, F as in the above problem.
    iterations: maximum number of iterations.
    tol: when the difference of solutions in two successive
        iterations less than this value, the algorithm will stop.

    Returns:
    --------
    """
    def calc_cost(D):
        return -2*np.trace(np.dot(E, D.T)) + np.trace(np.dot(np.dot(F, D.T), D))

    D_old = D.copy()
    for _ in range(iterations):
        for i in range(D.shape[1]):
            if F[i, i] != 0:
                a = 1.0/F[i, i] * (E[:, i] - D.dot(F[:, i])) + D[:, i]
                D[:, i] = a/max(LA.norm(a, 2), 1)

        if LA.norm(D - D_old, 'fro')/D.size < tol:
            break
        D_old = D.copy()
    return D


def DLSI_updateD(D, E, F, A, lambda1, iterations=100):
    """
    def DLSI_updateD(D, E, F, A, lambda1, verbose = False, iterations = 100):
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
    """
    def calc_cost(D):
        cost = -2*np.trace(np.dot(E, D.T)) + np.trace(np.dot(F, np.dot(D.T, D))) +\
            lambda1*utils.normF2(np.dot(A, D))
        return cost
    it = 0
    rho = 1.0
    Z_old = D.copy()
    U = np.zeros_like(D)
    I_k = np.eye(D.shape[1])
    X = 2*lambda1/rho*A.T
    Y = A.copy()
    B1 = np.dot(X, utils.inv_IpXY(Y, X))

    # B1 = np.dot(X, LA.inv(eye(Y.shape[0]) + np.dot(Y, X)))
    tol = 1e-8
    for it in range(iterations):
        it += 1
        # update D
        W  = Z_old - U
        E2 = E + rho/2*W
        F2 = F + rho/2*I_k
        D  = ODL_updateD(D, E2, F2)
        # update Z
        V = D + U
        Z_new = rho*(V - np.dot(B1, np.dot(Y, V)))
        e1 = utils.normF2(D - Z_new)
        e2 = rho*utils.normF2(Z_new - Z_old)
        if e1 < tol and e2 < tol:
            break
        U = U + D - Z_new
        Z_old = Z_new.copy()

    return D


def num_grad(func, X):
    """
    Calculating gradient of a function `func(X)` where `X` is a matrix or
    vector
    """
    grad = np.zeros_like(X)
    eps = 1e-4
    # TODO: flatten then unflatten, make it independent on X.shape
    # the current implementation only work with 2-d array
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # print X, '\n'
            Xp = X.copy()
            Xm = X.copy()
            Xp[i, j] += eps
            # print X
            fp = func(Xp)
            Xm[i, j] -= eps
            fm = func(Xm)
            grad[i, j] = (fp - fm)/(2*eps)
    return grad


def check_grad(func, grad, X):
    print('Checking grad...',)
    grad1 = grad(X)
    grad2 = num_grad(func, X)

    dif =  LA.norm(grad1 - grad2)
    if dif < 1e-5:
        print('Different = %f' %dif, 'PASS')
    else:
        print('Different = %f' %dif, 'FAIL')
    return dif < 1e-5


def min_rank_dict(Y, X, lambdaD, Dinit, iterations = 100, tol = 1e-8):
    """
    [D] = argmin_D 0.5*|| Y - DX||_F^2 + lambdaD ||D||_*
    s.t. ||d_i||_2^2 <= 1, for all i
    using ADMM:
    INPUT:
        Y: Data
        Dinit: intitial D
        X: sparse code
        lambdaD: regularization term
    OUTPUT:
        D:
    Created: Tiep Vu 6/29/2015 2:05:28 PM
    ------------------------
    Choose a rho.
    Algorithm summary
    ADMM: D,J = argmin_DJ 0.5*||Y - DX||_F^2 + lambdaD||J||_*
    s.t ||d_i||_2^2 <= 1 and J = D
    Alternatively solving:
    (1): D^{k+1} = argmin_D 0.5*||Y - DX||_F^2 + rho/2 ||J - D + U^k||_F^2
        s.t. ||d_i||_2^2 <= 1
        this problem can be soved using the update dictionary stage in
            Online Dictionary Learning method
    (2): J^{k+1} = argminJ lambdaD||J||_* + rho/2||J - D^{k+1} + U^k||
        Solution: shrinkage_rank(D^{k+1} - U^k, lambdaD/rho)
    (3): Update U: U^{k+1} = U^k + J^{k+1} - D^{k+1}
     Stoping cretia:
    ||r^k||_F^2 <= tol, ||s^k||_F^2 <= tol
    r^k = J^k - D^k
    s^k = rho(J^{k+1} - J^k)
    ---------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/22/2016
            http://www.personal.psu.edu/thv102/
    ---------------------------------------------
    """
    YXt = np.dot(Y, X.T)
    XXt = np.dot(X, X.T)
    rho = 0.25
    D_old = Dinit
    J_old = Dinit
    U_old = np.zeros_like(Dinit)
    it = 0
    I = np.eye(XXt.shape[0])
    tau = 2
    mu = 10.0
    for it in range(iterations):
        ## =========update D ================================
        # D = argmin_D 0.5*||Y - DX||_F^2 + rho/2 ||J - D + U||_F^2
        # s.t. ||d_i||_2^2 <= 1
        E = YXt + rho*(J_old + U_old)
        F = XXt + rho*I
        # D_new = updateD_EF(D_old, E, F, 10);
        D_new = ODL_updateD(D_old, E, F, iterations = 30)
        ## ========= update J ==============================
        # J^{k+1} = argminJ lambdaD||J||_* + rho/2||J - D + U||
        J_new = np.real(utils.shrinkage_rank(D_old - U_old, lambdaD/rho))
         ## ========= update U ==============================
        U_new = U_old + J_new - D_old
        ## ========= check stop ==============================
        r = J_new - D_old
        s = rho*(J_new - J_old)
        r_eps = LA.norm(r, 'fro')
        s_eps = LA.norm(s, 'fro')
        if r_eps < tol and s_eps < tol:
            break
        D_old = D_new
        J_old = J_new
        U_old = U_new
        if r_eps > mu*s_eps:
            rho = rho*tau
        elif s_eps > mu*r_eps:
            rho = rho/tau
    return D_new


class Fista(object):
    def __init__(self):
        """
        subclasses are required to have three following functions and lambd
        """
        self._grad = None
        self._calc_f = None
        self.lossF = None
        self.lambd = None
        self.D = None
        self.Y = None
        self.L = None

    def solve(self, Xinit=None, iterations=100, tol=1e-8, verbose=False):
        if Xinit is None:
            Xinit = np.zeros((self.D.shape[1], self.Y.shape[1]))
        Linv = 1/self.L
        lambdaLiv = self.lambd/self.L
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

    def _grad(self, y):
        raise NotImplementedError

    def lossF(self, x):
        raise NotImplementedError

    def check_grad(self, X):
        grad1 = self._grad(X)
        grad2 = num_grad(self._calc_f, X)
        dif = utils.norm1(grad1 - grad2)/grad1.size
        print('grad difference = %.7f'%dif)


class Lasso(Fista):
    """
    Solving a Lasso optimization problem using FISTA
    `X, = arg min_X 0.5*||Y - DX||_F^2 + lambd||X||_1
        = argmin_X f(X) + lambd||X||_1
        F(x) = f(X) + lamb||X||_1
    """
    def __init__(self, D, lambd = .1):
        self.D = D
        self.lambd = lambd
        self.DtD = np.dot(self.D.T, self.D)
        self.Y = None
        self.DtY = None
        self.L = np.max(LA.eig(self.DtD)[0])
        self.coef_ = None

    def fit(self, Y, Xinit = None, iterations = 100):
        self.Y = Y
        self.DtY = np.dot(self.D.T, self.Y)
        if Xinit is None:
            Xinit = np.zeros((self.D.shape[1], self.Y.shape[1]))
        self.coef_ = self.solve(Xinit=Xinit, iterations=iterations)

    def _grad(self, X):
        return np.dot(self.DtD, X) - self.DtY

    def _calc_f(self, X):
        return 0.5*utils.normF2(self.Y - np.dot(self.D, X))

    def lossF(self, X):
        return self._calc_f(X) + self.lambd*utils.norm1(X)