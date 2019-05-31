from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from constrained_cem_mpc import ConstrainedCemMpc
from constrained_cem_mpc.utils import assert_shape
from torch import Tensor

from .state_space_models import StateSpaceModel


def _dynamics_func(state, action):
    raise NotImplementedError


def _objective_func(states, actions):
    raise NotImplementedError


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
    def __init__(self, state_dimen: int):
        self._state_dimen = state_dimen

    def flatten(self, p: np.ndarray, q: Optional[np.ndarray]):
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


def _objective_func(states, actions):
    return 0


class CemSafeMPC:

    def __init__(self, state_dimen: int, action_dimen: int) -> None:
        super().__init__()
        self._mpc = ConstrainedCemMpc(_dynamics_func, _objective_func, constraints=None, state_dimen=state_dimen,
                                      action_dimen=action_dimen, time_horizon=5, num_rollouts=20, num_elites=3,
                                      num_iterations=1, num_workers=0)

    def init_solver(self, cost_func=None):
        pass

    def get_action(self, state: np.ndarray):
        # state: [state_dimen]
        raise NotImplementedError
