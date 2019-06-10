"""Contains the base class for Safe MPC implementations."""
from abc import ABC, abstractmethod
from typing import Tuple, Union, List

from numpy import ndarray


class SafeMPC(ABC):
    """Base class for implementations of the Safe MPC algorithm."""

    @property
    @abstractmethod
    def state_dimen(self) -> int:
        pass

    @property
    @abstractmethod
    def action_dimen(self) -> int:
        pass

    @property
    @abstractmethod
    def safety_trajectory_length(self) -> int:
        pass

    @property
    @abstractmethod
    def performance_trajectory_length(self) -> int:
        pass

    @property
    @abstractmethod
    def x_train(self) -> ndarray:
        """The x values of the current training data in the state space model."""
        pass

    @abstractmethod
    def information_gain(self) -> Union[ndarray, List[None]]:
        pass

    @abstractmethod
    def ssm_predict(self, z: ndarray) -> Tuple[ndarray, ndarray]:
        pass

    @abstractmethod
    def eval_prior(self, states: ndarray, actions: ndarray):
        """Compute the prediction of the prior model for the given state and action.

        :param states: [N x n_s] batch of N states
        :param actions: [N x n_u] batch of N actions
        :returns: [N x n_s] batch of N next states as predicted by the prior
        """
        pass

    @abstractmethod
    def init_solver(self, cost_func=None) -> None:
        pass

    @abstractmethod
    def get_action(self, state: ndarray) -> Tuple[ndarray, bool]:
        pass

    @abstractmethod
    def get_action_verbose(self, state: ndarray):
        pass

    @abstractmethod
    def update_model(self, x: ndarray, y: ndarray, opt_hyp=False, replace_old=True, reinitialize_solver=True) -> None:
        pass
