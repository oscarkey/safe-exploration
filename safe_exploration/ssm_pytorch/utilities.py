"""General utilities for pytorch."""

import itertools
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.autograd import grad

from ..utils import assert_shape

__all__ = ['compute_jacobian', 'update_cholesky', 'SetTorchDtype']


def compute_jacobian(f, x):
    """
    Compute the Jacobian matrix df/dx.

    Parameters
    ----------
    f : torch.Tensor
        The vector representing function values.
    x : torch.Tensor
        The vector with respect to which we want to take gradients.

    Returns
    -------
    df/dx : torch.Tensor
        A matrix of size f.size() + x.size() that contains the derivatives of
        the elements of f with respect to the elements of x.
    """
    assert x.requires_grad, 'Gradients of x must be required.'

    # Default to standard gradient in the 0d case
    if f.dim() == 0:
        return grad(f, x)[0]

    # Initialize outputs
    jacobian = torch.zeros(f.shape + x.shape, device=x.device)
    grad_output = torch.zeros(*f.shape, device=x.device)

    # Iterate over all elements in f
    for index in itertools.product(*map(range, f.shape)):
        grad_output[index] = 1
        g = grad(f, x, grad_outputs=grad_output, retain_graph=True, allow_unused=True)[0]
        # If there is no connection between f and x then g will be None, thus we leave it as 0.
        if g is not None:
            jacobian[index] = g
        grad_output[index] = 0

    return jacobian


def compute_jacobian_fast(f: Callable[[Tensor], Tensor], x: Tensor, num_outputs: int) -> Tensor:
    """Computes the jacobian of the output of f wrt x.

    Uses the batch support of f to compute all the elements of the jacobian in a single backward pass. Borrowed from:
    https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa

    :param f: [N x n] -> [N x m]
    :param x: [N x n], batch of inputs
    :param num_outputs: m, the output dimension of f
    :returns: [N x m x n], batch of jacobians
    """
    assert x.dim() == 2, f'Wanted [N x n], got {x.size()}'
    N, n = x.size()

    # For some reason, if requires_grad is True then grad is None. So we set it to false, and then true again below.
    x.requires_grad = False

    x = tile(x, num_outputs)
    x.requires_grad_(True)

    y = f(x)
    assert_shape(y, (N * num_outputs, num_outputs))

    repeated_eye = torch.eye(num_outputs, device=x.device).repeat((N, 1))
    y.backward(repeated_eye)

    if x.grad is None:
        # If there is no connection between f and x then g will be None, thus we leave it as 0.
        return torch.zeros((N, num_outputs, n), dtype=x.dtype)
    else:
        flat_jac = x.grad.data
        return flat_jac.view(N, num_outputs, n)


def tile(x: Tensor, n_tile: int):
    """Tiles Tensor x n_tile times in dimension 0.

    :param x: [N x n] input to tile
    :param n_tile: number of times x will be repeated in dim 0
    :returns: [N * n_tile x n]
    """
    assert x.dim() == 2
    assert n_tile >= 1
    return x.repeat(1, n_tile).view(-1, x.size(1))


class SetTorchDtype(object):
    """Context manager to temporarily change the pytorch dtype.

    Parameters
    ----------
    dtype : torch.dtype
    """

    def __init__(self, dtype):
        self.new_dtype = dtype
        self.old_dtype = None

    def __enter__(self):
        """Set new dtype."""
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.new_dtype)

    def __exit__(self, *args):
        """Restor old dtype."""
        torch.set_default_dtype(self.old_dtype)


def update_cholesky(old_chol, new_row, chol_row_out=None, jitter=1e-6):
    """Update an existing cholesky decomposition after adding a new row.

    TODO: Replace with fantasy data once this is available:
    https://github.com/cornellius-gp/gpytorch/issues/177

    A_new = [A, new_row[:-1, None],
             new_row[None, :]]

    old_chol = torch.cholesky(A, upper=False)

    Parameters
    ----------
    old_chol : torch.tensor
    new_row : torch.tensor
        1D array.
    chol_row_out : torch.tensor, optional
        An output array to which to write the new cholesky row.
    jitter : float
        The jitter to add to the last element of the new row. Makes everything
        numerically more stable.
    """
    new_row[-1] += jitter
    if len(new_row) == 1:
        if chol_row_out is not None:
            chol_row_out[:] = torch.sqrt(new_row[0])
            return
        else:
            return torch.sqrt(new_row.unsqueeze(-1))

    with SetTorchDtype(old_chol.dtype):
        c, _ = torch.trtrs(new_row[:-1], old_chol, upper=False)
        c = c.squeeze(-1)

        d = torch.sqrt(max(new_row[-1] - c.dot(c), torch.tensor(1e-10)))

        if chol_row_out is not None:
            chol_row_out[:-1] = c
            chol_row_out[-1] = d
        else:
            return torch.cat([
                torch.cat([old_chol, torch.zeros(old_chol.size(0), 1)], dim=1),
                torch.cat([c[None, :], d[None, None]], dim=1)
            ])
