from . import utils
from . import ODL
import numpy as np


def test_unit():
    print('\n===========================================================')
    print('Unit test: Online Dictionary Learning (ODL)')
    d = 10
    N = 50
    k = 20
    Y = utils.normc(np.random.randn(d, N))
    clf = ODL.ODL(k, lambd=0.01)
    clf.fit(Y, verbose=True, iterations=50)
