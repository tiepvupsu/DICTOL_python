import numpy as np 
import numpy.random as npr
import time
import theano.tensor as T
from theano import function


x = T.dmatrix('x')
y = T.dmatrix('y')
z = T.dot(x, y)
mydot = function([x, y], z)

N = 1
n = 1000
 
A = npr.randn(n,n)
B = npr.randn(n,n)
 
t = time.time()
for i in range(N):
    C = mydot(A, B)
td = time.time() - t
print("dotted two (%d,%d) matrices in %0.1f ms" % (n, n, 1e3*td/N))