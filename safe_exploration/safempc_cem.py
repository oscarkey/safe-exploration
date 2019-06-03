from abc import abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from constrained_cem_mpc import ConstrainedCemMpc, ActionConstraint, box2torchpoly, Constraint
from constrained_cem_mpc.utils import assert_shape
from polytope import Polytope
from torch import Tensor

from . import gp_reachability
from .environments import Environment
from .safempc_simple import LqrFeedbackController
from .state_space_models import StateSpaceModel
from .visualization import utils_visualization


class CemSSM(StateSpaceModel):
    """State Space Model interface for use with CEM MPC."""

    def __init__(self, state_dimen: int, action_dimen: int):
        super().__init__(state_dimen, action_dimen, has_jacobian=False, has_reverse=False)

    def predict(self, states, actions, jacobians=False, full_cov=False):
        if full_cov:
            raise ValueError('CEM MPC does not support full covariance at the moment')

        if jacobians:
            return self._predict_with_jacobians(states, actions)
        else:
            return self._predict_without_jacobians(states, actions)

    @abstractmethod
    def _predict_with_jacobians(self, states: np.ndarray, actions: np.ndarray) -> (
            np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        pass

    @abstractmethod
    def _predict_without_jacobians(self, states: np.ndarray, actions: np.ndarray) -> (np.ndarray, np.ndarray):
        pass

    def linearize_predict(self, states, actions, jacobians=False, full_cov=False):
        raise AttributeError('Not required for CEM MPC')

    def get_reverse(self, seed):
        raise AttributeError('Not required for CEM MPC')

    def get_linearize_reverse(self, seed):
        raise AttributeError('Not required for CEM MPC')

    def update_model(self, train_x, train_y, opt_hyp=False, replace_old=False):
        raise NotImplementedError

    def get_forward_model_casadi(self, linearize_mu=True):
        raise AttributeError('Not required for CEM MPC')


class FakeCemSSM(CemSSM):
    """Fake state space model for use during development."""

    def _predict(self, states: np.ndarray, actions: np.ndarray) -> (np.ndarray, np.ndarray):
        means = np.zeros_like(states)
        vares = np.ones_like(states)

        # Why do we transpose here?
        return means.T, vares.T

    def _predict_with_jacobians(self, states: np.ndarray, actions: np.ndarray) -> (
            np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        means, vares = self._predict(states, actions)
        jacobians = np.zeros((self.num_states, self.num_states + self.num_actions))
        return means, vares, jacobians

    def _predict_without_jacobians(self, states: np.ndarray, actions: np.ndarray) -> (np.ndarray, np.ndarray):
        return self._predict(states, actions)

    def update_model(self, train_x, train_y, opt_hyp=False, replace_old=False):
        raise NotImplementedError


class PQFlattener:
    """Hack which converts the p and q matrices which make up the state to a flat state vector, and vice versa.

    As each state is an ellipsoid it has both a centre (p) and a shape (q). However, constrained-cem-mpc currently
    requires the state to be a single vector, so this class handles the conversion between the two.
    """

    def __init__(self, state_dimen: int):
        self._state_dimen = state_dimen

    def flatten(self, p: np.ndarray, q: Optional[np.ndarray]) -> Tensor:
        if q is None:
            q = np.zeros((self._state_dimen, self._state_dimen))

        assert_shape(p, (self._state_dimen,))
        assert_shape(q, (self._state_dimen, self._state_dimen))
        flat = np.hstack((p.reshape(-1), q.reshape(-1)))
        return torch.tensor(flat)

    def unflatten(self, flat: Tensor) -> (np.ndarray, Optional[np.ndarray]):
        assert_shape(flat, (self.get_flat_state_dimen(),))
        p = flat[0:self._state_dimen].numpy()
        q = flat[self._state_dimen:].view(self._state_dimen, self._state_dimen).numpy()

        # If q is all zeros, we treat this as None.
        if not q.any():
            q = None

        return p, q

    def get_flat_state_dimen(self):
        return self._state_dimen + (self._state_dimen * self._state_dimen)


def _plot_constraints_in_2d(h_mat_safe, h_safe, h_mat_obs, h_obs):
    ax = plt.axes()
    Polytope(h_mat_obs[:, [0, 2]], h_obs).plot(ax=ax, color='blue')
    Polytope(h_mat_safe[:, [0, 2]], h_safe).plot(ax=ax, color='red')
    utils_visualization.plot_ellipsoid_2D(np.array([[0, 0]]).T, np.array([[0.1, 0.01], [0.01, 0.01]]), ax,
                                          color='green')
    ax.set_xticks(range(-12, 7))
    ax.set_xlabel('cart x position')
    ax.set_yticks(range(-1, 3))
    ax.set_ylabel('pendulum angle')


def _plot_ellipsoids_in_2d(p, q):
    utils_visualization.plot_ellipsoid_2D(p[[0, 2], :], np.delete(np.delete(q, [1, 2], 0), [1, 2], 1), ax=plt.gca(),
                                          color='orange', n_points=50)


class EllipsoidTerminalConstraint(Constraint):
    """Represents the terminal constraint of the MPC problem, for ellipsoid states and a polytopic constraint."""

    def __init__(self, state_dimen: int, safe_polytope_a, safe_polytope_b):
        self._pq_flattener = PQFlattener(state_dimen)
        self._polytope_a = safe_polytope_a
        self._polytope_b = safe_polytope_b

    def __call__(self, trajectory, actions) -> float:
        p, q = self._pq_flattener.unflatten(trajectory[-1])
        p = np.expand_dims(p, 1)
        if gp_reachability.is_ellipsoid_inside_polytope(p, q, self._polytope_a, self._polytope_b):
            return 0
        else:
            return 10


def _objective_func(states, actions):
    return 0


def construct_constraints(env: Environment):
    """Creates the polytopic constraints for the MPC problem from the values in the config file."""
    h_mat_safe, h_safe, h_mat_obs, h_obs = env.get_safety_constraints(normalize=True)
    return [ActionConstraint(box2torchpoly([[env.u_min.item(), env.u_max.item()], ])),  #
            EllipsoidTerminalConstraint(env.n_s, h_mat_safe, h_safe)]


class CemSafeMPC:
    """Safe MPC implementation which uses the constrained CEM to optimise the trajectories."""

    def __init__(self, state_dimen: int, action_dimen: int, constraints: [Constraint], opt_env, wx_feedback_cost,
                 wu_feedback_cost) -> None:
        super().__init__()
        self._state_dimen = state_dimen
        self._action_dimen = action_dimen
        self._pq_flattener = PQFlattener(state_dimen)
        self._mpc = ConstrainedCemMpc(self._dynamics_func, _objective_func, constraints=constraints,
                                      state_dimen=self._pq_flattener.get_flat_state_dimen(), action_dimen=action_dimen,
                                      time_horizon=1, num_rollouts=20, num_elites=3, num_iterations=10, num_workers=0)
        self._ssm = FakeCemSSM(state_dimen, action_dimen)

        # TODO: read l_mu and l_sigma from the config
        self._l_mu = np.array([0.05, 0.05, 0.05, 0.05])
        self._l_sigma = np.array([0.05, 0.05, 0.05, 0.05])

        self._linearized_model_a, self._linearized_model_b = opt_env['lin_model']
        self._lqr_controller = LqrFeedbackController(wx_feedback_cost, wu_feedback_cost, state_dimen, action_dimen,
                                                     self._linearized_model_a, self._linearized_model_b)

    def init_solver(self, cost_func=None):
        # TODO: attach the cost function to the mpc.
        pass

    def get_action(self, state: np.ndarray) -> (np.ndarray, bool):
        assert_shape(state, (self._state_dimen,))
        actions = self._mpc.get_actions(self._pq_flattener.flatten(state, None))
        # TODO: Need to maintain a queue of previously safe actions.
        if actions is not None:
            print('Found solution')
            success = True
            action = actions[0].numpy()
        else:
            print('No solution, using safe controller')
            success = False
            action = self._get_safe_controller_action(state)

        return action, success

    def _dynamics_func(self, state, action):
        p, q = self._pq_flattener.unflatten(state)
        p = np.expand_dims(p, 1)

        k_ff = np.expand_dims(action.numpy(), 1)
        p_next, q_next = gp_reachability.onestep_reachability(p, self._ssm, k_ff, self._l_mu, self._l_sigma, q,
                                                              k_fb=self._lqr_controller.get_control_matrix(),
                                                              a=self._linearized_model_a, b=self._linearized_model_b,
                                                              verbose=0)
        return self._pq_flattener.flatten(np.squeeze(p_next), q_next)

    def _get_safe_controller_action(self, state):
        # TODO: Use the safe policy from the config (though I think this is actually always just lqr)
        return np.dot(self._lqr_controller.get_control_matrix(), state)

    def update_model(self, x, y, opt_hyp=False, replace_old=True, reinitialize_solver=True):
        raise NotImplementedError
