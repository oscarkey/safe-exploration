# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:20:28 2017

@author: tkoller
"""
import numpy as np
import pytest

from ..utils import sample_inside_polytope, assert_shape


def test_sample_inside_polytope():
    """

    polytope:
        -.3 < x_1 < .4
        -.2 < x_2 < .2
    """
    x = np.array([[0.1, 0.15], [0.0, 0.0], [.5, .15]])

    a = np.vstack((np.eye(2), -np.eye(2), -np.eye(2)))
    b = np.array([.4, .2, .3, .2, .3, .2])[:, None]

    res = sample_inside_polytope(x, a, b)

    res_expect = np.array([True, True, False])  # should be: inside, inside, not inside

    assert np.all(
        res == res_expect), "Are the right samples inside/outside the polyope?"


def test___assert_shape__correct__does_nothing():
    assert_shape(np.zeros((10, 20, 30)), (10, 20, 30))


def test__assert_shape__incorrect__throws():
    with pytest.raises(ValueError):
        assert_shape(np.zeros((10, 20, 30)), (10, 20, 31))


def test__assert_shape__none_but_ignore_true__does_nothing():
    assert_shape(None, (), ignore_if_none=True)


def test__assert_shape__none_and_ignore_false__throws():
    with pytest.raises(ValueError):
        assert_shape(None, (), ignore_if_none=False)
