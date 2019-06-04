import os.path

import numpy as np
import pytest
import torch
from polytope import polytope

from .. import gp_reachability as gp_reachability_numpy
from .. import gp_reachability_pytorch
from ..safempc_cem import CemSSMNumpyWrapper
from ..ssm_cem import GpCemSSM


def setup_module():
    torch.set_default_dtype(torch.double)


@pytest.fixture(
    params=[("InvPend", True, True), ("InvPend", False, True), ("InvPend", True, True), ("InvPend", False, True)])
def before_test_onestep_reachability(request):
    np.random.seed(125)

    env, init_uncertainty, lin_model = request.param
    n_s = 2
    n_u = 1
    c_safety = 2
    a = None
    b = None
    if lin_model:
        a = np.random.rand(n_s, n_s)
        b = np.random.rand(n_s, n_u)

    train_data = np.load(os.path.join(os.path.dirname(__file__), 'invpend_data.npz'))
    X = torch.tensor(train_data["X"])
    y = torch.tensor(train_data["y"].T)

    ssm = GpCemSSM(n_s, n_u)
    ssm.update_model(X, y, replace_old=True)

    L_mu = np.array([0.001] * n_s)
    L_sigm = np.array([0.001] * n_s)
    k_fb = np.random.rand(n_u, n_s)  # need to choose this appropriately later
    k_ff = np.random.rand(n_u, 1)

    p = .1 * np.random.randn(n_s, 1)
    if init_uncertainty:
        q = .2 * np.array([[.5, .2], [.2, .65]])  # reachability based on previous uncertainty
    else:
        q = None  # no initial uncertainty

    return n_s, n_u, p, q, ssm, k_fb, k_ff, L_mu, L_sigm, c_safety, a, b


def test__onestep_reachability__returns_the_same_as_numpy_impl(before_test_onestep_reachability):
    n_s, n_u, p, q, ssm, k_fb, k_ff, L_mu, L_sigm, c_safety, a, b = before_test_onestep_reachability
    numpy_ssm = CemSSMNumpyWrapper(n_s, n_u, ssm)

    p_numpy, q_numpy = gp_reachability_numpy.onestep_reachability(p, numpy_ssm, k_ff, L_mu, L_sigm, q, k_fb, c_safety,
                                                                  verbose=0, a=a, b=b)

    q = torch.tensor(q) if q is not None else q
    p_pytorch, q_pytorch = gp_reachability_pytorch.onestep_reachability(torch.tensor(p), ssm, torch.tensor(k_ff),
                                                                        torch.tensor(L_mu), torch.tensor(L_sigm), q,
                                                                        torch.tensor(k_fb), c_safety, verbose=0,
                                                                        a=torch.tensor(a), b=torch.tensor(b))

    assert np.allclose(p_pytorch.detach().numpy(), p_numpy), "Centers of the next states should be the same"
    assert np.allclose(q_pytorch.detach().numpy(), q_numpy), "Shapes of the next states should be the same"


def test__is_ellipsoid_inside_polytope__inside__returns_true():
    poly = polytope.box2poly(([[0., 10.], [0., 10.]]))
    A = torch.tensor(poly.A)
    b = torch.tensor(poly.b).unsqueeze(1)
    p = torch.tensor([[5., 5.]]).transpose(0, 1)
    q = torch.tensor([[2., 1.], [1., 2.]])
    assert gp_reachability_pytorch.is_ellipsoid_inside_polytope(p, q, A, b) is True


def test__is_ellipsoid_inside_polytope__partially_out__returns_false():
    poly = polytope.box2poly(([[0., 10.], [0., 10.]]))
    A = torch.tensor(poly.A)
    b = torch.tensor(poly.b).unsqueeze(1)
    p = torch.tensor([[0., 0.]]).transpose(0, 1)
    q = torch.tensor([[2., 1.], [1., 2.]])
    assert gp_reachability_pytorch.is_ellipsoid_inside_polytope(p, q, A, b) is False


def test__is_ellipsoid_inside_polytope__outside__returns_false():
    poly = polytope.box2poly(([[0., 10.], [0., 10.]]))
    A = torch.tensor(poly.A)
    b = torch.tensor(poly.b).unsqueeze(1)
    p = torch.tensor([[20., 20.]]).transpose(0, 1)
    q = torch.tensor([[2., 1.], [1., 2.]])
    assert gp_reachability_pytorch.is_ellipsoid_inside_polytope(p, q, A, b) is False
