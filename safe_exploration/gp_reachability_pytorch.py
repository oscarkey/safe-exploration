"""Contains functions for computing ellipsoid trajectories, and ellipsoid polytope intersection.

This is a copy of gp_reachability, but converted to PyTorch. Thus it works more efficiently with CemSafeMPC, which also
uses PyTorch.
"""
from typing import Tuple

import torch
from torch import Tensor

from .ssm_cem import CemSSM
from .utils import print_ellipsoid, assert_shape, compute_remainder_overapproximations_pytorch
from .utils_ellipsoid import ellipsoid_from_rectangle_pytorch, sum_two_ellipsoids_pytorch


def onestep_reachability(p_center: Tensor, ssm: CemSSM, k_ff: Tensor, l_mu: Tensor, l_sigma: Tensor,
                         q_shape: Tensor = None, k_fb: Tensor = None, c_safety: float = 1., verbose: int = 1,
                         a: Tensor = None, b: Tensor = None) -> Tuple[Tensor, Tensor]:
    """Over-approximate the reachable set of states under affine control law.

    Given a system of the form:
        x_{t+1} = \mathcal{N}(\mu(x_t,u_t), \Sigma(x_t,u_t)),
    where x,\mu \in R^{n_s}, u \in R^{n_u} and \Sigma^{n_s \times n_s} are given by the gp predictive mean and variance
    respectively, we approximate the reachset of a set of inputs x_t \in \epsilon(p,Q) describing an ellipsoid with
    center p and shape matrix Q under the control low u_t = Kx_t + k

    :param p_center: [n_s x 1] Center of the state ellipsoid
    :param q_shape: [n_s x n_s] Shape of the state ellipsoid
    :param ssm: The state space model of the dynamics
    :param k_ff: [n_u x 1] The feedforward control matrix
    :param l_mu: [n_s]
    :param l_sigma: [n_s]
    :param k_fb: [n_u x n_s]
    :param c_safety: The scaling of the semi-axes of the uncertainty matrix corresponding to a level-set of the gaussian
     pdf.
    :param verbose: Verbosity level of the print output
    :param a: Parameter of the linear model.
    :param b: Parameter of the linear model.
    :returns: ([n_s x 1] Center of the overapproximated next state ellipsoid,
    [n_s x n_s] Shape matrix of the overapproximated next state ellipsoid)
    """
    assert_shape(p_center, (ssm.num_states, 1))
    assert_shape(k_ff, (1, ssm.num_actions))
    assert_shape(l_mu, (ssm.num_states,))
    assert_shape(l_sigma, (ssm.num_states,))
    assert_shape(q_shape, (ssm.num_states, ssm.num_states), ignore_if_none=True)
    assert_shape(k_fb, (1, ssm.num_states), ignore_if_none=True)
    assert_shape(a, (ssm.num_states, ssm.num_states), ignore_if_none=True)
    assert_shape(b, (ssm.num_states, 1), ignore_if_none=True)

    n_s = p_center.shape[0]
    n_u = k_ff.shape[0]

    if a is None:
        a = torch.eye(n_s)
        b = torch.zeros((n_s, n_u))

    if q_shape is None:
        # The state is a point.
        u_p = k_ff

        if verbose > 0:
            print("\nApplying action:")
            print(u_p)

        mu_0, sigm_0 = ssm.predict_without_jacobians(p_center.transpose(0, 1), u_p.transpose(0, 1))

        rkhs_bounds = c_safety * torch.sqrt(sigm_0.transpose(0, 1)).view((n_s,))

        q_1 = ellipsoid_from_rectangle_pytorch(rkhs_bounds)

        p_lin = torch.mm(a, p_center) + torch.mm(b, u_p)
        p_1 = p_lin + mu_0

        if verbose > 0:
            print_ellipsoid(p_1, q_1, text="uncertainty first state")

        return p_1.detach(), q_1.detach()
    else:
        # The state is a (ellipsoid) set.
        if verbose > 0:
            print_ellipsoid(p_center, q_shape, text="initial uncertainty ellipsoid")
        # compute the linearization centers
        x_bar = p_center  # center of the state ellipsoid
        # u_bar = k_fb*(u_bar-u_bar) + k_ff = k_ff
        u_bar = k_ff

        if verbose > 0:
            print("\nApplying action:")
            print(u_bar)
        # compute the zero and first order matrices
        mu_0, sigm_0, jac_mu = ssm.predict_with_jacobians(x_bar.transpose(0, 1), u_bar.transpose(0, 1))

        if verbose > 0:
            print_ellipsoid(mu_0, torch.diag(sigm_0.squeeze()), text="predictive distribution")

        a_mu = jac_mu[:, :n_s]
        b_mu = jac_mu[:, n_s:]

        # reach set of the affine terms
        H = a + a_mu + torch.mm(b_mu + b, k_fb)
        p_0 = mu_0 + torch.mm(a, x_bar) + torch.mm(b, u_bar)

        Q_0 = torch.mm(H, torch.mm(q_shape, H.transpose(0, 1)))

        if verbose > 0:
            print_ellipsoid(p_0, Q_0, text="linear transformation uncertainty")

        # computing the box approximate to the lagrange remainder
        ub_mean, ub_sigma = compute_remainder_overapproximations_pytorch(q_shape, k_fb, l_mu, l_sigma)
        b_sigma_eps = c_safety * (torch.sqrt(sigm_0.transpose(0, 1)) + ub_sigma)

        Q_lagrange_sigm = ellipsoid_from_rectangle_pytorch(b_sigma_eps.squeeze())
        p_lagrange_sigm = torch.zeros((n_s, 1))

        if verbose > 0:
            print_ellipsoid(p_lagrange_sigm, Q_lagrange_sigm, text="overapproximation lagrangian sigma")

        Q_lagrange_mu = ellipsoid_from_rectangle_pytorch(ub_mean)
        p_lagrange_mu = torch.zeros((n_s, 1))

        if verbose > 0:
            print_ellipsoid(p_lagrange_mu, Q_lagrange_mu, text="overapproximation lagrangian mu")

        p_sum_lagrange, Q_sum_lagrange = sum_two_ellipsoids_pytorch(p_lagrange_sigm, Q_lagrange_sigm, p_lagrange_mu,
                                                                    Q_lagrange_mu)

        p_1, q_1 = sum_two_ellipsoids_pytorch(p_sum_lagrange, Q_sum_lagrange, p_0, Q_0)

        if verbose > 0:
            print_ellipsoid(p_1, q_1, text="accumulated uncertainty current step")

            print("volume of ellipsoid summed individually")
            print((torch.det(torch.cholesky(q_1))))

        return p_1.detach(), q_1.detach()


def lin_ellipsoid_safety_distance(p_center: Tensor, q_shape: Tensor, h_mat: Tensor, h_vec: Tensor,
                                  c_safety: float = 1.0) -> Tensor:
    """Compute the distance between ellipsoid and polytope

    Evaluate the distance of an  ellipsoid E(p_center,q_shape), to a polytopic set
    of the form:
        h_mat * x <= h_vec.

    :param p_center: [n_s x 1] The center of the state ellipsoid
    :param q_shape: [n_s x n_s] The shape matrix of the state ellipsoid
    :param h_mat: [m x n_s] The shape matrix of the safe polytope (see above)
    :param h_vec: [m x 1] The additive vector of the safe polytope (see above)

    :returns: d_safety: [m] The distance of the ellipsoid to the polytope. If d < 0 (elementwise), the ellipsoid is
    inside the poltyope (safe), otherwise safety is not guaranteed.
    """
    m, n_s = h_mat.shape
    assert_shape(p_center, (n_s, 1))
    assert_shape(q_shape, (n_s, n_s))
    assert_shape(h_vec, (m, 1))

    d_center = torch.mm(h_mat, p_center)
    # MISSING SQRT (?)
    d_shape = c_safety * torch.sqrt(
        torch.sum(torch.mm(q_shape, h_mat.transpose(0, 1)) * h_mat.transpose(0, 1), dim=0)[:, None])
    d_safety = d_center + d_shape - h_vec

    return d_safety


def is_ellipsoid_inside_polytope(p_center: Tensor, q_shape: Tensor, h_mat: Tensor, h_vec: Tensor) -> bool:
    """Returns True if the ellipsoid with center p and shape q is inside the polytope, otherwise False.

    The polytope is of the form  h_mat * x <= h_vec.
    """
    d_safety = lin_ellipsoid_safety_distance(p_center, q_shape, h_mat, h_vec)
    # The ellipsoid is safely inside if none of the values of d_safety are >= 0.
    return not (d_safety >= 0).any()
