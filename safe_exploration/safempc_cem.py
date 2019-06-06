from typing import Optional, Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from constrained_cem_mpc import ConstrainedCemMpc, ActionConstraint, box2torchpoly, Constraint, Rollout, DynamicsFunc
from constrained_cem_mpc.utils import assert_shape
from numpy import ndarray
from polytope import Polytope
from torch import Tensor

from . import gp_reachability_pytorch
from .environments import Environment
from .gp_reachability_pytorch import onestep_reachability
from .safempc_simple import LqrFeedbackController
from .ssm_cem import GpCemSSM
from .visualization import utils_visualization


class PQFlattener:
    """Converts ellipsoid p and q matrices to a flat state vector, and vice versa.

    As each state is an ellipsoid it has both a centre (p) and a shape (q). However, constrained-cem-mpc currently
    requires the state to be a single vector, so this class handles the conversion between the two.
    """

    def __init__(self, state_dimen: int):
        self._state_dimen = state_dimen

    def flatten(self, p: Tensor, q: Optional[Tensor]) -> Tensor:
        if q is None:
            q = torch.zeros((self._state_dimen, self._state_dimen), dtype=p.dtype, device=p.device)

        assert_shape(p, (self._state_dimen,))
        assert_shape(q, (self._state_dimen, self._state_dimen))
        return torch.cat((p.reshape(-1), q.reshape(-1)))

    def unflatten(self, flat: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        assert_shape(flat, (self.get_flat_state_dimen(),))
        p = flat[0:self._state_dimen]
        q = flat[self._state_dimen:].view(self._state_dimen, self._state_dimen)

        # If q is all zeros, we treat this as None.
        if len(q.nonzero()) == 0:
            q = None

        return p, q

    def get_flat_state_dimen(self):
        return self._state_dimen + (self._state_dimen * self._state_dimen)


def _plot_constraints_in_2d(h_mat_safe, h_safe, h_mat_obs, h_obs) -> None:
    ax = plt.axes()
    # Hacky way to test if cartpole or pendulum.
    num_cartpole_states = 4
    if h_mat_safe.shape[1] == num_cartpole_states:
        Polytope(h_mat_obs[:, [0, 2]], h_obs).plot(ax=ax, color='blue')
        Polytope(h_mat_safe[:, [0, 2]], h_safe).plot(ax=ax, color='red')
    else:
        Polytope(h_mat_safe, h_safe).plot(ax=ax, color='grey')


def _plot_ellipsoids_in_2d(p: Tensor, q: Tensor, color: Union[str, Tuple[float, float, float]] = 'orange') -> None:
    p = p.detach().numpy()
    q = q.detach().numpy()
    # Hacky way to test if cartpole or pendulum.
    num_cartpole_states = 4
    if p.shape[0] == num_cartpole_states:
        utils_visualization.plot_ellipsoid_2D(p[[0, 2], :], np.delete(np.delete(q, [1, 2], 0), [1, 2], 1), ax=plt.gca(),
                                              color=color, n_points=50)
    else:
        utils_visualization.plot_ellipsoid_2D(p, q, ax=plt.gca(), color=color, n_points=50)


class EllipsoidTerminalConstraint(Constraint):
    """Represents the terminal constraint of the MPC problem, for ellipsoid states and a polytopic constraint."""

    def __init__(self, state_dimen: int, safe_polytope_a: np.ndarray, safe_polytope_b: np.ndarray):
        self._pq_flattener = PQFlattener(state_dimen)
        self._polytope_a = torch.tensor(safe_polytope_a)
        self._polytope_b = torch.tensor(safe_polytope_b)

    def __call__(self, trajectory, actions) -> float:
        p, q = self._pq_flattener.unflatten(trajectory[-1])
        p = p.unsqueeze(1)
        if gp_reachability_pytorch.is_ellipsoid_inside_polytope(p, q, self._polytope_a, self._polytope_b):
            return 0
        else:
            return 10


def construct_constraints(env: Environment):
    """Creates the polytopic constraints for the MPC problem from the values in the config file."""
    h_mat_safe, h_safe, h_mat_obs, h_obs = env.get_safety_constraints(normalize=True)
    return [ActionConstraint(box2torchpoly([[env.u_min.item(), env.u_max.item()], ])),  #
            EllipsoidTerminalConstraint(env.n_s, h_mat_safe, h_safe)]


class DynamicsFuncWrapper(DynamicsFunc):
    """Wraps a given function as a DynamicsFunc."""

    def __init__(self, func):
        self._func = func

    def __call__(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        return self._func(state, action)


class CemSafeMPC:
    """Safe MPC implementation which uses the constrained CEM to optimise the trajectories."""

    def __init__(self, constraints: [Constraint], env: Environment, opt_env, wx_feedback_cost,
                 wu_feedback_cost) -> None:
        super().__init__()

        self._state_dimen = env.n_s
        self._action_dimen = env.n_u
        self._l_mu = torch.tensor(env.l_mu)
        self._l_sigma = torch.tensor(env.l_sigm)
        self._get_random_action = env.random_action
        self._pq_flattener = PQFlattener(env.n_s)
        self._ssm = GpCemSSM(env.n_s, env.n_u)

        linearized_model_a, linearized_model_b = opt_env['lin_model']
        self.lin_model = opt_env['lin_model']
        self._linearized_model_a = torch.tensor(linearized_model_a)
        self._linearized_model_b = torch.tensor(linearized_model_b)
        self._lqr = LqrFeedbackController(wx_feedback_cost, wu_feedback_cost, env.n_s, env.n_u, linearized_model_a,
                                          linearized_model_b)

        # TODO: Load params for CEM from config.
        self._mpc = ConstrainedCemMpc(DynamicsFuncWrapper(self._dynamics_func), constraints=constraints,
                                      state_dimen=self._pq_flattener.get_flat_state_dimen(), action_dimen=env.n_u,
                                      time_horizon=2, num_rollouts=20, num_elites=3, num_iterations=8, num_workers=0)

        self._has_training_data = False

    def init_solver(self, cost_func=None):
        # TODO: attach the cost function to the mpc.
        pass

    def get_action(self, state: ndarray) -> Tuple[ndarray, bool]:
        assert_shape(state, (self._state_dimen,))

        # If we don't have training data we skip solving the mpc as it won't be any use.
        # This makes the first episode much faster (during which we gather training data).
        if self._has_training_data:
            actions, rollouts = self._mpc.get_actions(self._pq_flattener.flatten(torch.tensor(state), None))
            self._plot_rollouts(rollouts)
        else:
            actions = None

        # TODO: Need to maintain a queue of previously safe actions to fully implement SafeMPC.

        if actions is not None:
            print('Found solution')
            success = True
            action = actions[0].numpy()
        elif np.random.rand() < 0.2:
            # This is a temporary epsilon-greedy like exploration policy to help gather data for the GP.
            # Obviously taking random actions isn't actually safe, but it's useful for debugging.
            print('No solution, taking random action')
            success = False
            action = self._get_random_action()
        else:
            print('No solution, using safe controller')
            success = False
            action = self._get_safe_controller_action(state)

        return action, success

    def _plot_rollouts(self, rollouts: List[List[Rollout]]):
        """Plots the constraint, and the terminal states of the rollouts through the optimisation process."""
        tc = self._mpc._rollout_function._constraints[1]
        _plot_constraints_in_2d(tc._polytope_a.detach().numpy(), tc._polytope_b.detach().numpy(), None, None)
        for i in range(len(rollouts)):
            for j in range(len(rollouts[i])):
                t = rollouts[i][j].trajectory[-1]
                p, q = self._pq_flattener.unflatten(t)
                red = hex(round(255.0 / self._mpc._num_iterations * i))[2:]
                if red == '0':
                    red = '00'
                color = f'#{red}0000'
                _plot_ellipsoids_in_2d(p.unsqueeze(1), q, color=color)

        plt.show()

    def _dynamics_func(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        p, q = self._pq_flattener.unflatten(state)
        p = p.unsqueeze(1)

        k_ff = action.unsqueeze(1)
        p_next, q_next, sigma = onestep_reachability(p, self._ssm, k_ff, self._l_mu, self._l_sigma, q,
                                                     k_fb=self._lqr.get_control_matrix_pytorch(),
                                                     a=self._linearized_model_a, b=self._linearized_model_b, verbose=0)
        return self._pq_flattener.flatten(p_next.squeeze(), q_next), torch.tensor([0.0])

    def _get_safe_controller_action(self, state):
        # TODO: Use the safe policy from the config (though I think this is actually always just lqr)
        return np.dot(self._lqr.get_control_matrix(), state)

    def update_model(self, x: ndarray, y: ndarray, opt_hyp=False, replace_old=True, reinitialize_solver=True) -> None:
        self._ssm.update_model(torch.tensor(x), torch.tensor(y), opt_hyp, replace_old)
        self._has_training_data = True
