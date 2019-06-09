# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:22:17 2017

@author: tkoller
"""
import numpy as np
import pytest
import torch

from ..utils_ellipsoid import distance_to_center, ellipsoid_from_rectangle, ellipsoid_from_rectangle_pytorch, \
    sum_two_ellipsoids, sum_two_ellipsoids_pytorch


def setup_module():
    torch.set_default_dtype(torch.double)


def test__sum_two_ellipsoids_pytorch__gives_same_result_as_numpy_impl():
    p_a_1 = torch.tensor([0.0, 1.0])
    q_a_1 = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    p_b_1 = torch.tensor([2.0, 2.0])
    q_b_1 = torch.tensor([[4.0, 1.0], [1.0, 2.0]])

    p_a_2 = torch.tensor([10.1, 11.1])
    q_a_2 = torch.tensor([[12.1, 11.1], [11.1, 12.1]])
    p_b_2 = torch.tensor([12.1, 12.1])
    q_b_2 = torch.tensor([[14.1, 11.1], [11.1, 12.1]])

    p_a_3 = torch.tensor([100.1, 101.1])
    q_a_3 = torch.tensor([[102.1, 101.1], [101.1, 102.1]])
    p_b_3 = torch.tensor([102.1, 102.1])
    q_b_3 = torch.tensor([[104.1, 101.1], [101.1, 102.1]])

    p_a_batch = torch.stack((p_a_1, p_a_2, p_a_3))
    q_a_batch = torch.stack((q_a_1, q_a_2, q_a_3))
    p_b_batch = torch.stack((p_b_1, p_b_2, p_b_3))
    q_b_batch = torch.stack((q_b_1, q_b_2, q_b_3))

    p_numpy_1, q_numpy_1 = sum_two_ellipsoids(p_a_1.numpy(), q_a_1.numpy(), p_b_1.numpy(), q_b_1.numpy())
    p_numpy_2, q_numpy_2 = sum_two_ellipsoids(p_a_2.numpy(), q_a_2.numpy(), p_b_2.numpy(), q_b_2.numpy())
    p_numpy_3, q_numpy_3 = sum_two_ellipsoids(p_a_3.numpy(), q_a_3.numpy(), p_b_3.numpy(), q_b_3.numpy())
    p_torch, q_torch = sum_two_ellipsoids_pytorch(p_a_batch, q_a_batch, p_b_batch, q_b_batch)

    assert np.allclose(p_numpy_1, p_torch[0].numpy())
    assert np.allclose(q_numpy_1, q_torch[0].numpy())
    assert np.allclose(p_numpy_2, p_torch[1].numpy())
    assert np.allclose(q_numpy_2, q_torch[1].numpy())
    assert np.allclose(p_numpy_3, p_torch[2].numpy())
    assert np.allclose(q_numpy_3, q_torch[2].numpy())


@pytest.fixture(params=["rectangle", "cube"])
def before_ellipsoid_from_rectangle(request):
    if request.param == "rectangle":
        n_s = 3
        ub = [0.1, 0.3, 0.5]
        test_points = np.array([[-0.1, -0.3, 0.5], [-0.1, 0.3, -0.5], [0.1, 0.3, 0.5]])

    else:
        n_s = 3
        ub = [0.1] * n_s
        test_points = np.array([[-0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])
    test_data = {"ub": ub, "n_s": n_s, "test_points": test_points}
    return test_data


def test_ellipsoid_from_rectangle_ub_below_zero_throws_exception():
    """ do we get an exception if lb > ub """

    with pytest.raises(Exception):
        ub = [0.6, -0.3]
        q_shape = ellipsoid_from_rectangle(ub)


def test_ellipsoid_from_from_rectangle_shape_matrix_spd(before_ellipsoid_from_rectangle):
    """ Is the resulting shape matrix symmetric postive definite?"""

    ub = before_ellipsoid_from_rectangle["ub"]
    n_s = before_ellipsoid_from_rectangle["n_s"]

    q_shape = ellipsoid_from_rectangle(ub)

    assert np.all(np.linalg.eigvals(q_shape) > 0)
    assert np.allclose(0.5 * (q_shape + q_shape.T), q_shape)


def test_ellipsoid_from_from_rectangle_residuals_zero_(before_ellipsoid_from_rectangle):
    """ Are the residuals of the exact algebraic fit zero at the edges of the rectangle? """
    eps_tol = 1e-5

    ub = before_ellipsoid_from_rectangle["ub"]
    n_s = before_ellipsoid_from_rectangle["n_s"]

    q_shape = ellipsoid_from_rectangle(ub)

    p_center = np.zeros((n_s, 1))

    test_points = before_ellipsoid_from_rectangle["test_points"]

    d_test_points = distance_to_center(test_points, p_center, q_shape)

    assert np.all(np.abs(d_test_points - 1) <= eps_tol)


def test__ellipsoid_from_rectangle_pytorch__ub_below_zero__throws_exception():
    with pytest.raises(Exception):
        ub = torch.tensor([[0.6, -0.3]])
        ellipsoid_from_rectangle_pytorch(ub)


def test__ellipsoid_from_from_rectangle_pytorch__returns_same_as_numpy_impl():
    ub1 = torch.tensor([0.5, 0.2])
    ub2 = torch.tensor([0.3, 10.0])
    ub3 = torch.tensor([0.1, 5.6])
    ub_batch = torch.stack((ub1, ub2, ub3))

    q_numpy_1 = ellipsoid_from_rectangle(ub1.numpy())
    q_numpy_2 = ellipsoid_from_rectangle(ub2.numpy())
    q_numpy_3 = ellipsoid_from_rectangle(ub3.numpy())
    q_pytorch = ellipsoid_from_rectangle_pytorch(ub_batch)

    assert np.allclose(q_numpy_1, q_pytorch[0].numpy())
    assert np.allclose(q_numpy_2, q_pytorch[1].numpy())
    assert np.allclose(q_numpy_3, q_pytorch[2].numpy())


@pytest.fixture(params=["t_1", "t_2", "t_3"])
def before_distance_tests(request):
    test_name = request.param
    if test_name == "t_1":

        q = 4 * np.eye(2)
        p = np.eye(2, 1)
        x = np.eye(2, 1) + np.sqrt(2)
        d = 1
    elif test_name == "t_2":
        q = 4 * np.eye(2)
        p = np.zeros((2, 1))
        x = np.zeros((2, 1)) + np.sqrt(2)
        d = 1
    elif test_name == "t_3":
        q = 4 * np.eye(2)
        p = np.eye(2, 1)
        x = np.eye(2, 1) + np.sqrt(2)
        d = 1

    return p, q, x.T, d


def test_distance_to_center(before_distance_tests):
    """ """
    p, q, x, d = before_distance_tests

    assert np.allclose(distance_to_center(x, p, q), d)
