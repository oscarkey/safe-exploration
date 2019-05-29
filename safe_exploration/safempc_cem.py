from abc import abstractmethod

import numpy as np
from constrained_cem_mpc import ConstrainedCemMpc

from .state_space_models import StateSpaceModel


def _dynamics_func(state, action):
    raise NotImplementedError


def _objective_func(states, actions):
    raise NotImplementedError


class CemSSM(StateSpaceModel):
    """State Space Model interface for use with CEM MPC."""

    def predict(self, states, actions, jacobians=False, full_cov=False):
        if jacobians:
            raise ValueError('CEM MPC does not require/support jacobians')
        self._predict(states, actions, full_cov)

    @abstractmethod
    def _predict(self, states: np.ndarray, actions: np.ndarray, full_cov: bool):
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

    def _predict(self, states: np.ndarray, actions: np.ndarray, full_cov: bool) -> (np.ndarray, np.ndarray):
        raise NotImplementedError

    def update_model(self, train_x, train_y, opt_hyp=False, replace_old=False):
        raise NotImplementedError


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
