import numpy as np
from . import utils
from qcore.asserts import assert_eq


class TestLabelToRange(object):
    def test_1(self):
        label = [1, 1, 1, 2, 2, 2, 2, 3, 3]
        got_range = utils.label_to_range(label)
        expected_range = [0, 3, 7, 9]
        assert_eq(expected_range, got_range)

    def test_2(self):
        label = [1, 2, 3]
        got_range = utils.label_to_range(label)
        expected_range = [0, 1, 2, 3]
        assert_eq(expected_range, got_range)

    def test_raise_if_not_consecutive_1(self):
        # raise error if incorrect input format
        pass


class TestRangeToLabel(object):
    def test_1(self):
        a_range = [0, 3, 5, 9]
        label = utils.range_to_label(a_range)
        expected_label = [1, 1, 1, 2, 2, 3, 3, 3, 3]
        assert_eq(expected_label, label)

    def test_2(self):
        a_range = [0, 1]
        label = utils.range_to_label(a_range)
        expected_label = [1]
        assert_eq(expected_label, label)

    def test_3(self):
        # raise error if incorrect input format
        pass



class TestGetBlockCol(object):
    def test_1(self):
        matrix = np.array([[1, 2, 3, 4, 5],
                           [1, 2, 3, 4, 5]])
        block_indices = 1
        col_range = [0, 2, 4, 5]
        expected_array = np.array([[3, 4],
                                   [3, 4]])
        got_array = utils.get_block_col(matrix, block_indices, col_range)
        np.testing.assert_array_equal(expected_array, got_array)

        block_indices = [0, 2]
        expected_array = np.array([[1, 2, 5],
                                   [1, 2, 5]])
        got_array = utils.get_block_col(matrix, block_indices, col_range)
        np.testing.assert_array_equal(expected_array, got_array)


class TestGetBlockRow(object):
    def test_1(self):
        matrix = np.array([[1, 1],
                           [2, 2],
                           [3, 3],
                           [4, 4],
                           [5, 5],
                           [6, 6]])
        block_indices = 1
        row_range = [0, 2, 5, 6]
        expected_array = np.array([[3, 3],
                                   [4, 4],
                                   [5, 5]])
        got_array = utils.get_block_row(matrix, block_indices, row_range)
        np.testing.assert_array_equal(expected_array, got_array)

        block_indices = [0, 2]
        expected_array = np.array([[1, 1],
                                   [2, 2],
                                   [6, 6]])
        got_array = utils.get_block_row(matrix, block_indices, row_range)
        np.testing.assert_array_equal(expected_array, got_array)



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
