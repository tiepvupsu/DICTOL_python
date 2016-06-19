from utils import *
from ODL import *
def min_rank_dict0(Y, X, lambdaD, Dinit, opts):
    """
    This function try to solve the following problem:
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
    optsD = opts 
    optsD.max_iter = 50
    while it < opts.max_iter:
        it += 1 
        ## =========update D ================================
        # D = argmin_D 0.5*||Y - DX||_F^2 + rho/2 ||J - D + U||_F^2 
        # s.t. ||d_i||_2^2 <= 1
        E = YXt + rho*(J_old + U_old)
        F = XXt + rho*I
        # D_new = updateD_EF(D_old, E, F, 10);
        D_new = ODL_updateD(D_old, E, F, optsD)[0]
        ## ========= update J ==============================
        # J^{k+1} = argminJ lambdaD||J||_* + rho/2||J - D + U||
        J_new = np.real(shrinkage_rank(D_new - U_old, lambdaD/rho))

        ## ========= update U ==============================
        U_new = U_old + J_new - D_new
        
        ## ========= check stop ==============================
        r = J_new - D_new
        s = rho*(J_new - J_old)
        r_eps = LA.norm(r, 'fro')
        s_eps = LA.norm(s, 'fro')   
        if r_eps < opts.tol and s_eps < opts.tol:
            break
        D_old = D_new
        J_old = J_new
        U_old = U_new
        if r_eps > mu*s_eps:
            rho = rho*tau
        elif s_eps > mu*r_eps:
            rho = rho/tau

    return D_new 