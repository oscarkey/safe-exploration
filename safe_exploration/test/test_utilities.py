"""Test the functions in utilities.py"""

import pytest

from .test_ssm_cem import TestGpCemSSM
from ..ssm_cem.ssm_cem import GpCemSSM
from ..ssm_pytorch.utilities import compute_jacobian_fast, tile

try:
    import torch
    from torch import Tensor
    from safe_exploration.ssm_pytorch import compute_jacobian, update_cholesky, SetTorchDtype
except:
    pass



@pytest.fixture(autouse = True)
def check_has_ssm_pytorch_module(check_has_ssm_pytorch):
    pass


class TestJacobian(object):

    def test_error(self):
        """Test assertion error raised when grad is missing."""
        with pytest.raises(AssertionError):
            compute_jacobian(None, torch.ones(2, 1))

    def test_0d(self):
        x = torch.ones(2, 2, requires_grad=True)
        A = torch.tensor([[1., 2.], [3., 4.]])
        f = A * x
        f = torch.sum(f)

        jac = compute_jacobian(f, x)
        torch.testing.assert_allclose(jac, A)

    def test_1d(self):
        """Test jacobian function for 1D inputs."""
        x = torch.ones(1, requires_grad=True)
        f = 2 * x

        jac = compute_jacobian(f, x)
        torch.testing.assert_allclose(jac[0, 0], 2)

    def test_2d(self):
        """Test jacobian computation."""
        x = torch.ones(2, 1, requires_grad=True)
        A = torch.tensor([[1., 2.], [3., 4.]])
        f = A @ x

        jac = compute_jacobian(f, x)
        torch.testing.assert_allclose(A, jac[:, 0, :, 0])

        # Test both multiple runs
        jac = compute_jacobian(f.squeeze(-1), x)
        torch.testing.assert_allclose(A, jac.squeeze(-1))

    def test_2d_output(self):
        """Test jacobian with 2d input and output"""
        x = torch.ones(2, 2, requires_grad=True)
        A = torch.tensor([[1., 2.], [3., 4.]])
        f = A * x

        jac = compute_jacobian(f, x)
        torch.testing.assert_allclose(jac.shape, 2)
        torch.testing.assert_allclose(jac.sum(dim=0).sum(dim=0), A)


def test__compute_jacobian_fast__returns_same_as_slow_impl():
    ssm = GpCemSSM(TestGpCemSSM.FakeConfig, state_dimen=2, action_dimen=1)
    train_x = torch.tensor([[1., 2., 3.]])
    train_y = torch.tensor([[10., 11.]])
    ssm.update_model(train_x, train_y, replace_old=True)

    z = torch.tensor([[1., 2., 4.], [0.5, 2.5, 3.2], [0.4, 2.7, 3.8], [0.2, 2.9, 3.9]])
    z.requires_grad = True

    def f(x: Tensor):
        return ssm.predict_raw(x)[1].transpose(0, 1)

    fast_jac = compute_jacobian_fast(f, z, num_outputs=2)

    z1 = torch.tensor([1., 2., 4.])
    z2 = torch.tensor([0.5, 2.5, 3.2])
    z3 = torch.tensor([0.4, 2.7, 3.8])
    z4 = torch.tensor([0.2, 2.9, 3.9])
    z1.requires_grad = True
    z2.requires_grad = True
    z3.requires_grad = True
    z4.requires_grad = True
    y1 = ssm.predict_raw(z1.unsqueeze(0))[1].squeeze()
    y2 = ssm.predict_raw(z2.unsqueeze(0))[1].squeeze()
    y3 = ssm.predict_raw(z3.unsqueeze(0))[1].squeeze()
    y4 = ssm.predict_raw(z4.unsqueeze(0))[1].squeeze()
    slow_jac1 = compute_jacobian(y1, z1)
    slow_jac2 = compute_jacobian(y2, z2)
    slow_jac3 = compute_jacobian(y3, z3)
    slow_jac4 = compute_jacobian(y4, z4)

    assert torch.allclose(fast_jac[0], slow_jac1)
    assert torch.allclose(fast_jac[1], slow_jac2)
    assert torch.allclose(fast_jac[2], slow_jac3)
    assert torch.allclose(fast_jac[3], slow_jac4)


def test__compute_jacobian_fast__f_does_not_use_z__returns_zero():
    N = 10
    n_s = 5
    n_u = 2
    z = torch.ones((N, n_s + n_u))

    def f(x: Tensor):
        return torch.ones((x.size(0), n_s), requires_grad=True)

    fast_jac = compute_jacobian_fast(f, z, num_outputs=n_s)

    assert fast_jac.nonzero().size(0) == 0


def test__tile__only_once__returns_input_unchanged():
    x = torch.tensor([[1, 2], [4, 5], [7, 8]])
    tiled = tile(x, 1)
    assert torch.allclose(tiled, x)


def test__tile__more_than_once__returns_tiled_result():
    x = torch.tensor([[1, 2 , 3], [4, 5, 6]])

    tiled = tile(x, 3)

    expected = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6]])
    assert torch.allclose(tiled, expected)


def test_update_cholesky():
    """Test that the update cholesky function returns correct values."""
    n = 6
    new_A = torch.rand(n, n, dtype=torch.float64)
    new_A = new_A @ new_A.t()
    new_A += torch.eye(len(new_A), dtype=torch.float64)

    A = new_A[:n - 1, :n - 1]

    old_chol = torch.cholesky(A, upper=False)
    new_row = new_A[-1]

    # Test updateing overall
    new_chol = update_cholesky(old_chol, new_row)
    error = new_chol - torch.cholesky(new_A, upper=False)
    assert torch.all(torch.abs(error) <= 1e-15)

    # Test updating inplace
    new_chol = torch.zeros(n, n, dtype=torch.float64)
    new_chol[:n - 1, :n - 1] = old_chol

    update_cholesky(old_chol, new_row, chol_row_out=new_chol[-1])
    error = new_chol - torch.cholesky(new_A, upper=False)
    assert torch.all(torch.abs(error) <= 1e-15)


def test_set_torch_dtype():
    """Test dtype context manager."""
    dtype = torch.get_default_dtype()

    torch.set_default_dtype(torch.float32)
    with SetTorchDtype(torch.float64):
        a = torch.zeros(1)

    assert a.dtype is torch.float64
    b = torch.zeros(1)
    assert b.dtype is torch.float32

    torch.set_default_dtype(dtype)
