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
