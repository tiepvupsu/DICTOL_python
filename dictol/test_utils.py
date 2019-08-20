import numpy as np
from . import utils
from qcore.asserts import assert_eq


def test_label_to_range():
    label = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3])
    got_range = utils.label_to_range(label)
    expected_range = [0, 3, 7, 9]
    np.testing.assert_array_equal(expected_range, got_range)


def test_range_to_label():
    arange = np.array([0, 3, 5, 9])
    label = utils.range_to_label(arange)
    expected_label = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    np.testing.assert_array_equal(expected_label, label)


def test_get_block_col():
    # TODO
    pass


def test_get_block_row():
    # TODO
    pass


def test_get_block():
    # TODO
    pass


def test_norm1():
    # TODO
    pass


def test_normF2():
    # TODO
    pass



def test_shrinkage():
    # TODO
    pass


def test_shrinkage_rank():
    # TODO
    pass


def MyForm_test():
    # TODO: uncomment these lines
    pass
    # print('-------------------------------------------')
    # print('`MyForm` test:')
    # d = 20
    # k = 50
    # M = np.random.randint(10, size = (d, d))
    # N = np.random.randint(10, size = (d, d))
    # N = np.zeros_like(M)
    # # M = np.zeros_like(N)
    # # print N
    # A = np.random.randint(10, size = (d, d))
    # B = np.random.randint(10, size = (d, d))

    # P = MyForm(M, N, k)
    # Q = MyForm(A, B, k)

    # # Multiplication test
    # print('1. Multiplication test...', end=' ')
    # dif = LA.norm(np.dot(Q.full_express(),P.full_express()) - \
    #     Q.mult(P).full_express())
    # if dif < 1e-8:
    #     print('diff =', dif, '\n   ...PASS')
    # else:
    #     print('diff =', dif, '\n   ...FAIL')

    # # Inverse test
    # print('2. Inverse test ...', end=' ')
    # X = LA.inv(P.full_express())
    # Y = P.inv()
    # dif = LA.norm(X - Y.full_express())
    # if dif < 1e-8:
    #     print('diff =', dif, '\n   ...PASS')
    # else:
    #     print('diff =', dif, '\n   ...FAIL')

    # # vector multiplication
    # print('3. Multiplication with vector...', end=' ')
    # x = np.random.randint(3, size =(d*k,))
    # # print x
    # # print P.full_express()
    # y = np.dot(P.full_express(), x)
    # z = P.mult_vec(x)

    # # print 'true\n', y
    # # print 'computed\n', z
    # dif = LA.norm(y - z)
    # if dif < 1e-8:
    #     print('diff =', dif, '\n   ...PASS')
    # else:
    #     print('diff =', dif, '\n   ...FAIL')


def test_pickDfromY():
    # TODO:
    pass
    # print ('----------------------------------')
    # print('Test `pickDfromY`......')
    # d = 2
    # n = 10
    # Y = np.random.randint(10, size=(d, n))
    # print(Y)
    # Y_range = np.array([0, 4, 10])
    # D_range = np.array([0, 2, 5])
    # D = pickDfromY(Y, Y_range, D_range)
    # # print Y_range
    # print(D)


def range_reduce_test():
    D_range = np.array([0, 4, 8, 13])
    bad_ids = np.array([1, 5, 7, 9, 10])
    print(D_range, bad_ids)
    range_reduce(D_range, bad_ids)
    print(D_range)


def test_inv_IpXY():
    # TODO: untodo this
    pass
    #d1 = 1000
    #d2 = 10
    #X = np.random.rand(d1, d2)
    #Y = np.random.rand(d2, d1)
    #t1 = time.time()
    #A = LA.inv(np.eye(d1)+ np.dot(X,Y))
    #t2 = time.time()
    #print('t1 = ', t2 - t1)
    ##
    #t1 = time.time()
    #B = inv_IpXY(X, Y)
    #t2 = time.time()
    #print('t2 = ', t2 - t1)
    #print('diff = ', normF2(A - B))