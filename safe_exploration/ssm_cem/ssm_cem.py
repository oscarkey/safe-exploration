"""Contains state space models for use with CemSafeMPC. These should all using PyTorch."""
from abc import abstractmethod, ABC
from typing import Tuple, Optional, Dict

import torch
from torch import Tensor

from ..utils import assert_shape


class CemSSM(ABC):
    """State space model interface for use with CEM MPC.

    Unlike StateSpaceModel it uses Tensors rather than NumPy arrays. It also does not specify the methods required only
    for Casadi.
    """

    def __init__(self, num_states: int, num_actions: int):
        self.num_states = num_states
        self.num_actions = num_actions

        self._x_train = None
        self._y_train = None

    @property
    def x_train(self) -> Optional[Tensor]:
        """Returns the x values of the current training data."""
        return self._x_train

    @property
    def y_train(self) -> Optional[Tensor]:
        return self._y_train

    @abstractmethod
    def predict_with_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict the next states and uncertainties, along with the jacobian of the state means.

        N is the batch size.

        :param states: (N x n_s) tensor of states
        :param actions: (N x n_u) tensor of actions
        :returns:
            mean of next states [N x n_s],
            variance of next states [N x n_s],
            jacobian of mean [N x n_s x n_s])
        """
        pass

    @abstractmethod
    def predict_without_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict the next states and uncertainties.

        N is the batck size.

        :param states (N x n_s) tensor of states
        :param actions (N x n_u) tensor of actions
        :returns:
            mean of next states [N x n_s],
            variance of next states [N x n_s]
        """
        pass

    @abstractmethod
    def predict_raw(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict using stacked state and action.

        :param z: [1 x (n_s + n_u)]
        :returns: [1 x n_s], [1 x n_s]
        """
        pass

    def update_model(self, train_x: Tensor, train_y: Tensor, opt_hyp=False, replace_old=False) -> None:
        """Incorporate the given data into the model.

        :param train_x: [N x (n_s + n_u)]
        :param train_y: [N x n_s]
        :param opt_hyp: if True we will train the model, which may take some time
        :param replace_old: If True, replace all existing training data with this new training data. Otherwise, merge
                            it.
        """
        N = train_x.size(0)
        assert_shape(train_x, (N, self.num_states + self.num_actions))
        assert_shape(train_y, (N, self.num_states))

        # TODO: select datapoints by highest variance.
        if replace_old or self._x_train is None or self._y_train is None:
            x_new = train_x
            y_new = train_y
        else:
            x_new = torch.cat((self._x_train, train_x), dim=0)
            y_new = torch.cat((self._y_train, train_y), dim=0)

        self._x_train = x_new
        self._y_train = y_new

        self._update_model(x_new, y_new)

        if opt_hyp:
            self._train_model(x_new, y_new)

    @abstractmethod
    def _update_model(self, x_train: Tensor, y_train: Tensor) -> None:
        """Subclasses should implement this to update the actual model, if needed."""
        pass

    @abstractmethod
    def _train_model(self, x_train: Tensor, y_train: Tensor) -> None:
        """Subclasses should implement this to train their model."""
        pass

    def _join_states_actions(self, states: Tensor, actions: Tensor) -> Tensor:
        N = states.size(0)
        assert_shape(states, (N, self.num_states))
        assert_shape(actions, (N, self.num_actions))
        return torch.cat((states, actions), dim=1)

    @abstractmethod
    def collect_metrics(self) -> Dict[str, float]:
        """Returns interesting metrics for the current state of the model.

        :returns: pairs (metric key, metric value) to log
        """
        pass
