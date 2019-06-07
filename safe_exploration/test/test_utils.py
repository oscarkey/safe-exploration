# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:20:28 2017

@author: tkoller
"""
import numpy as np
import pytest
import torch

from ..utils import sample_inside_polytope, assert_shape, compute_remainder_overapproximations, \
    compute_remainder_overapproximations_pytorch, eigenvalues_batch, trace_batch, batch_vector_matrix_mul, \
    batch_vector_mul


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


def test__compute_remainder_overapproximations_pytorch__returns_same_as_numpy_impl():
    q_1 = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    k_fb_1 = torch.tensor([[1.0, 5.0], [5.0, 9.0]])
    q_2 = torch.tensor([[10.0, 9.0], [0.5, 2.9]])
    k_fb_2 = torch.tensor([[1.1, 5.2], [4.5, 6.5]])
    q_batch = torch.stack((q_1, q_2))
    k_fb_batch = torch.stack((k_fb_1, k_fb_2))


    l_mu = torch.tensor([0.01, 0.05])
    l_sigma = torch.tensor([0.02, 0.03])
    l_mu_batch = l_mu.repeat((2, 1))
    l_sigma_batch = l_sigma.repeat((2, 1))

    u_numpy_1, sigma_numpy_1 = compute_remainder_overapproximations(q_1.numpy(), k_fb_1.numpy(), l_mu.numpy(),
                                                                    l_sigma.numpy())
    u_numpy_2, sigma_numpy_2 = compute_remainder_overapproximations(q_2.numpy(), k_fb_2.numpy(), l_mu.numpy(),
                                                                    l_sigma.numpy())
    u_pytorch, sigma_pytorch = compute_remainder_overapproximations_pytorch(q_batch, k_fb_batch, l_mu_batch,
                                                                            l_sigma_batch)

    assert np.allclose(u_numpy_1, u_pytorch[0].numpy())
    assert np.allclose(sigma_numpy_1, sigma_pytorch[0].numpy())
    assert np.allclose(u_numpy_2, u_pytorch[1].numpy())
    assert np.allclose(sigma_numpy_2, sigma_pytorch[1].numpy())


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


def test__eigenvalues_batch__returns_same_result_as_torch():
    x1 = torch.tensor([[1., 2.], [3., 4.]])
    x2 = torch.tensor([[10., 11.], [12., 13.]])
    x3 = torch.tensor([[100., 110.], [120., 130.]])
    x_batch = torch.stack((x1, x2, x3))

    evs_batch = eigenvalues_batch(x_batch)

    evs1, _ = torch.eig(x1)
    evs2, _ = torch.eig(x2)
    evs3, _ = torch.eig(x3)
    assert torch.allclose(evs_batch[0], evs1)
    assert torch.allclose(evs_batch[1], evs2)
    assert torch.allclose(evs_batch[2], evs3)


def test__trace_batch__returns_same_result_as_torch():
    x1 = torch.tensor([[1., 2.], [3., 4.]])
    x2 = torch.tensor([[10., 11.], [12., 13.]])
    x3 = torch.tensor([[100., 110.], [120., 130.]])
    x_batch = torch.stack((x1, x2, x3))

    traces_batch = trace_batch(x_batch)

    trace1 = torch.trace(x1)
    trace2 = torch.trace(x2)
    trace3 = torch.trace(x3)
    assert torch.allclose(traces_batch[0], trace1)
    assert torch.allclose(traces_batch[1], trace2)
    assert torch.allclose(traces_batch[2], trace3)


def test__batch_vector_tensor_mul__returns_same_as_individual_multiplications():
    x = torch.tensor([[1, 2], [3, 4]])
    v1 = torch.tensor([1, 2])
    v2 = torch.tensor([10, 20])
    v3 = torch.tensor([100, 200])
    v_batch = torch.stack((v1, v2, v3))

    r_batch = batch_vector_matrix_mul(x, v_batch)

    assert r_batch.size() == (3, 2)
    assert torch.allclose(r_batch[0], torch.matmul(x, v1))
    assert torch.allclose(r_batch[1], torch.matmul(x, v2))
    assert torch.allclose(r_batch[2], torch.matmul(x, v3))


def test__batch_vector_mul__returns_same_as_individual_multiplications():
    x = torch.tensor([[1, 2]])
    print('x shape', x.shape)
    v1 = torch.tensor([1, 2])
    v2 = torch.tensor([10, 20])
    v3 = torch.tensor([100, 200])
    v_batch = torch.stack((v1, v2, v3))

    r_batch = batch_vector_mul(x, v_batch)

    assert r_batch.size() == (3, 2)
    print('shape1', x.shape, v1.shape)
    print('shape2', torch.matmul(x, v1).shape)
    assert torch.allclose(r_batch[0], torch.matmul(x, v1))
    assert torch.allclose(r_batch[1], torch.matmul(x, v2))
    assert torch.allclose(r_batch[2], torch.matmul(x, v3))
