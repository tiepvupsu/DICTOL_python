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
