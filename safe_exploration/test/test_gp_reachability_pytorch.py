import os.path

import numpy as np
import pytest
import torch
from polytope import polytope

from .test_ssm_cem import TestGpCemSSM
from utils import assert_shape
from .. import gp_reachability as reachability_np
from .. import gp_reachability_pytorch as reachability_pt
from ..ssm_cem import CemSSM
from ..ssm_cem import GpCemSSM
from ..state_space_models import StateSpaceModel


def setup_module():
    torch.set_default_dtype(torch.double)


class CemSSMNumpyWrapper(StateSpaceModel):
    """Wrapper around CemSSMs which allows them to work with NumPy code."""

    def __init__(self, state_dimen: int, action_dimen: int, ssm: CemSSM):
        super().__init__(state_dimen, action_dimen)
        self._ssm = ssm

    def predict(self, states, actions, jacobians=False, full_cov=False):
        if full_cov:
            raise NotImplementedError

        if jacobians:
            p, sigma, j = self._ssm.predict_with_jacobians(torch.tensor(states), torch.tensor(actions))
            # CemSSM returns p: [N x n_s], StateSpaceModel returns [N x n_s x 1] (not sure why).
            p = p.unsqueeze(1)
            return self._convert_to_numpy((p, sigma, j))
        else:
            p, sigma = self._ssm.predict_without_jacobians(torch.tensor(states), torch.tensor(actions))
            # CemSSM returns p: [N x n_s], StateSpaceModel returns [N x n_s x 1] (not sure why).
            p = p.unsqueeze(1)
            return self._convert_to_numpy(p, sigma)

    @staticmethod
    def _convert_to_numpy(xs):
        return [x.detach().numpy() for x in xs]

    def linearize_predict(self, states, actions, jacobians=False, full_cov=False):
        raise NotImplementedError

    def get_reverse(self, seed):
        raise NotImplementedError

    def get_linearize_reverse(self, seed):
        raise NotImplementedError

    def update_model(self, train_x, train_y, opt_hyp=False, replace_old=False):
        self._ssm.update_model(torch.tensor(train_x), torch.tensor(train_y), opt_hyp, replace_old)


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
        a = torch.rand((n_s, n_s))
        b = torch.rand((n_s, n_u))
    a_np = a.numpy() if a is not None else a
    b_np = b.numpy() if b is not None else b

    train_data = np.load(os.path.join(os.path.dirname(__file__), 'invpend_data.npz'))
    X = torch.tensor(train_data["X"])
    y = torch.tensor(train_data["y"])

    ssm = GpCemSSM(TestGpCemSSM.FakeConfig(), n_s, n_u)
    ssm.update_model(X, y, replace_old=True)

    L_mu = torch.tensor([0.001] * n_s)
    L_sigm = torch.tensor([0.001] * n_s)
    k_fb = torch.rand((n_u, n_s))  # need to choose this appropriately later

    return n_s, n_u, ssm, k_fb, L_mu, L_sigm, c_safety, a, b, a_np, b_np, init_uncertainty


def test__onestep_reachability__returns_the_same_as_numpy_impl(before_test_onestep_reachability):
    N = 2
    n_s, n_u, ssm, k_fb, L_mu, L_sigm, c_safety, a, b, a_np, b_np, init_uncertainty = before_test_onestep_reachability
    numpy_ssm = CemSSMNumpyWrapper(n_s, n_u, ssm)

    p = .1 * torch.rand((N, n_s))
    if init_uncertainty:
        q = .2 * torch.tensor([[[.6, .21], [.21, .55]], [[.5, .2], [.2, .65]]])
        assert_shape(q, (N, n_s, n_s))
    else:
        q = None
    k_ff = torch.rand((N, n_u))

    # Remove batches.
    p_1 = p[0].unsqueeze(1).numpy()
    k_ff_1 = k_ff[0].unsqueeze(0).numpy()
    q_1 = q[0].numpy() if q is not None else q
    p_2 = p[1].unsqueeze(1).numpy()
    k_ff_2 = k_ff[1].unsqueeze(0).numpy()
    q_2 = q[1].numpy() if q is not None else q
    p_new_1, q_new_1 = reachability_np.onestep_reachability(p_1, numpy_ssm, k_ff_1, L_mu.numpy(), L_sigm.numpy(), q_1,
                                                            k_fb.numpy(), c_safety, verbose=0, a=a_np, b=b_np)
    p_new_2, q_new_2 = reachability_np.onestep_reachability(p_2, numpy_ssm, k_ff_2, L_mu.numpy(), L_sigm.numpy(), q_2,
                                                            k_fb.numpy(), c_safety, verbose=0, a=a_np, b=b_np)

    p_new_pt, q_new_pt, _ = reachability_pt.onestep_reachability(p, ssm, k_ff, L_mu, L_sigm, q, k_fb, c_safety,
                                                                 verbose=0, a=a, b=b)

    assert np.allclose(p_new_pt[0].numpy(), p_new_1.squeeze())
    assert np.allclose(q_new_pt[0].numpy(), q_new_1.squeeze())
    assert np.allclose(p_new_pt[1].numpy(), p_new_2.squeeze())
    assert np.allclose(q_new_pt[1].numpy(), q_new_2.squeeze())


def test__onestep_reachability__sigma_same_as_gp_output(before_test_onestep_reachability):
    N = 2
    n_s, n_u, ssm, k_fb, L_mu, L_sigm, c_safety, a, b, a_np, b_np, init_uncertainty = before_test_onestep_reachability

    p = .1 * torch.rand((N, n_s))
    if init_uncertainty:
        q = .2 * torch.tensor([[[.6, .21], [.21, .55]], [[.5, .2], [.2, .65]]])
        assert_shape(q, (N, n_s, n_s))
    else:
        q = None
    k_ff = torch.rand((N, n_u))

    _, _, pred_sigma = reachability_pt.onestep_reachability(p, ssm, k_ff, L_mu, L_sigm, q, k_fb, c_safety, verbose=0,
                                                            a=a, b=b)

    _, actual_sigma_1 = ssm.predict_without_jacobians(p[0:1], k_ff[0:1])
    _, actual_sigma_2 = ssm.predict_without_jacobians(p[1:2], k_ff[1:2])

    assert torch.allclose(pred_sigma[0], actual_sigma_1)
    assert torch.allclose(pred_sigma[1], actual_sigma_2)
    assert pred_sigma.shape == (N, n_s)


def test__is_ellipsoid_inside_polytope__inside__returns_true():
    poly = polytope.box2poly(([[0., 10.], [0., 10.]]))
    A = torch.tensor(poly.A)
    b = torch.tensor(poly.b).unsqueeze(1)
    p = torch.tensor([[5., 5.]]).transpose(0, 1)
    q = torch.tensor([[2., 1.], [1., 2.]])
    assert reachability_pt.is_ellipsoid_inside_polytope(p, q, A, b) is True


def test__is_ellipsoid_inside_polytope__partially_out__returns_false():
    poly = polytope.box2poly(([[0., 10.], [0., 10.]]))
    A = torch.tensor(poly.A)
    b = torch.tensor(poly.b).unsqueeze(1)
    p = torch.tensor([[0., 0.]]).transpose(0, 1)
    q = torch.tensor([[2., 1.], [1., 2.]])
    assert reachability_pt.is_ellipsoid_inside_polytope(p, q, A, b) is False


def test__is_ellipsoid_inside_polytope__outside__returns_false():
    poly = polytope.box2poly(([[0., 10.], [0., 10.]]))
    A = torch.tensor(poly.A)
    b = torch.tensor(poly.b).unsqueeze(1)
    p = torch.tensor([[20., 20.]]).transpose(0, 1)
    q = torch.tensor([[2., 1.], [1., 2.]])
    assert reachability_pt.is_ellipsoid_inside_polytope(p, q, A, b) is False
