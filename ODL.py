from utils import * 

# inv_IpXY_test()
def ODL_cost(Y, D, X, lambda1):
	"""
	cost = 0.5* ||Y - DX||_F^2 + lambda1||X||_1
	"""
	return  0.5*normF2(Y - np.dot(D, X)) + lambda1*norm1(X)

def ODL_updateD(D, E, F, opts):
	"""
	* The main algorithm in ODL. 
	* Solving the optimization problem:
	  `D = arg min_D -2trace(E'*D) + trace(D*F*D')` subject to: `||d_i||_2 <= 1`,
	     where `F` is a positive semidefinite matrix. 
	* Syntax `[D, iter] = ODL_updateD(D, E, F, opts)`
	  - INPUT: 
	    + `D, E, F` as in the above problem.
	    + `opts`. options:
	      * `opts.max_iter`: maximum number of iterations.
	      * `opts.tol`: when the difference between `D` in two successive 
			iterations less than this value, the algorithm will stop.
	  - OUTPUT:
	    + `D`: solution.
	    + `iter`: number of run iterations.
	-----------------------------------------------
	Author: Tiep Vu, thv102@psu.edu, 04/07/2016
	        (http://www.personal.psu.edu/thv102/)
	-----------------------------------------------
	"""
	D_old = D.copy() 
	it = 0 
	k = D.shape[1] 
	while it < opts.max_iter:
		it = it + 1 
		for i in xrange(k):
			if F[i,i] != 0:
				a = (E[:, i] - D.dot(F[:, i])/F[i,i] + D[:, i])
				D[:,i] = a/max(LA.norm(a, 2), 1)
		if LA.norm(D - D_old, 'fro') < opts.tol:
			break 
		D_old = D 
	return (D, it) 

def ODL_main(Y, k, lambda1, opts, method = 'fista'):
	"""
	* Solving the following problem:
	 (D, X) = \arg\min_{D,X} 0.5||Y - DX||_F^2 + lambda1||X||_1
	* Syntax: `(D, X) = ODL_main(Y, k, lambda1, opts)`
	  - INPUT: 
	    + `Y`: collection of samples.4/7/2016 7:35:39 PM
	    + `k`: number of atoms in the desired dictionary.
	    + `lambda1`: norm 1 regularization parameter.
	    + `opts`: option.
	    + `sc_method`: sparse coding method used in the sparse coefficient update. Possible values:
	      * `'fista'`: using FISTA algorithm. See also [`fista`](#fista).
	      * `'spams'`: using SPAMS toolbox [[12]](#fn_spams). 
	  - OUTPUT:
	    + `D, X`: as in the problem.
	-----------------------------------------------
	Author: Tiep Vu, thv102@psu.edu, 4/7/2016
	        (http://www.personal.psu.edu/thv102/)
	-----------------------------------------------
	"""
	Y_range = np.array([0, Y.shape[1]])
	D_range = np.array([0, k])
	D = pickDfromY(Y, Y_range, D_range)
	X = np.zeros((D.shape[1], Y.shape[1]))
	print 'Initial cost: %5.4f' % ODL_cost(Y, D, X, lambda1) 
	it = 0 
	optsX = Opts(max_iter = 300)
	optsD = Opts(max_iter = 200, tol = 1e-8)
	while it < opts.max_iter:
		it += 1 
		# Sparse coding 
		if method == 'fista':
			X, itx = lasso_fista(Y, D, X, lambda1, optsX)
		if opts.show_cost: 
			costX = ODL_cost(Y, D, X, lambda1)
			print 'iter: %3d' % it, 'costX = %4.4f' % costX, 'it: ', itx
		# Dictionary update 
		F = np.dot(X, X.T)
		E = np.dot(Y, X.T) 
		D, itd = ODL_updateD(D, E, F, optsD)
		if opts.show_cost:
			costD = ODL_cost(Y, D, X, lambda1)
			print 'iter: %3d' % it, 'costD = %4.4f' % costD, 'it: ', itd
			if abs(costX - costD) < opts.tol:
				break 
	print 'Final cost: %4.4f' % ODL_cost(Y, D, X, lambda1)
	return (D, X)		

def ODL_test():
	d      = 50 # data dimension
	N      = 10 # number of samples 
	k      = 5 # dictionary size 
	lambda1 = 0.1
	Y      = normc(np.random.rand(d, N))
	D      = normc(np.random.rand(d, k))
	# Xinit  = np.zeros(D.shape[1], Y.shape[1])
	opts = Opts(show_cost = True, max_iter = 100)
	ODL_main(Y, k, lambda1, opts)

ODL_test()

