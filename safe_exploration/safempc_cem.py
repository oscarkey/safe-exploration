from abc import abstractmethod

import numpy as np

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
