"""Contains state space models for use with CemSafeMPC. These should all using PyTorch."""
from abc import abstractmethod, ABC
from typing import Tuple, Optional, Dict, Callable, Any

import torch
from torch import Tensor

from ..utils import assert_shape


class CemSSM(ABC):
    """State space model interface for use with CEM MPC.

    Unlike StateSpaceModel it uses Tensors rather than NumPy arrays. It also does not specify the methods required only
    for Casadi.
    """

    def __init__(self, state_dimen: int, action_dimen: int):
        self.num_states = state_dimen
        self.num_actions = action_dimen

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
            jacobian of mean wrt state and action [N x n_s x (n_s + n_u)])
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

        if self.parametric or opt_hyp:
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
    def collect_metrics(self) -> Dict[str, Any]:
        """Returns interesting metrics for the current state of the model.

        The metrics may not be scalar, thus should be logged as non-scalar values.

        :returns: pairs (metric key, metric value) to log
        """
        pass

    @property
    @abstractmethod
    def parametric(self) -> bool:
        """Whether this is a parametric or non-parametric model."""
        pass


class JunkDimensionsSSM(CemSSM):
    """Wraps an SSM, adding a set of junk dimensions to every call.

    This is a convenient way to see how the ssm copes in a higher dimension environment, without actually implementing
    a higher dimensional environment.
    """

    def __init__(self, constructor: Callable[[int, int], CemSSM], state_dimen: int, action_dimen: int, junk_states: int,
                 junk_actions: int):
        """Constructs a new instance wrapping the environment created by the given constructor.

        :param constructor: partially evaluated constructor of the environment to wrap, which takes args state_dimen and
         action_dimen
        :param state_dimen: number of true states
        :param action_dimen: number of true actions
        :param junk_states: number of junk dimensions to add to state
        :param junk_actions: number of junk dimensions to add to action
        """
        super().__init__(state_dimen, action_dimen)
        total_states = state_dimen + junk_states
        total_actions = action_dimen + junk_actions
        self._junk_states = junk_states
        self._junk_actions = junk_actions
        self._ssm = constructor(state_dimen=total_states, action_dimen=total_actions)

    def predict_with_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        expanded_states = self._expand(states, self.num_states, self._junk_states)
        expanded_actions = self._expand(actions, self.num_actions, self._junk_actions)
        expanded_means, expanded_vars, expanded_jacs = self._ssm.predict_with_jacobians(expanded_states,
                                                                                        expanded_actions)
        return (expanded_means[:, :self.num_states], expanded_vars[:, :self.num_states],
                expanded_jacs[:, :self.num_states, :(self.num_states + self.num_actions)])

    def predict_without_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        expanded_states = self._expand(states, self.num_states, self._junk_states)
        expanded_actions = self._expand(actions, self.num_actions, self._junk_actions)
        expanded_means, expanded_vars = self._ssm.predict_without_jacobians(expanded_states, expanded_actions)
        return expanded_means[:, :self.num_states], expanded_vars[:, :self.num_states]

    def predict_raw(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        expanded_z = self._expand(z, self.num_states + self.num_actions, self._junk_states + self._junk_actions)
        expanded_means, expanded_vars = self._ssm.predict_raw(expanded_z)
        return expanded_means[:, :self.num_states], expanded_vars[:, :self.num_states]

    def update_model(self, train_x: Tensor, train_y: Tensor, opt_hyp=False, replace_old=False) -> None:
        super().update_model(train_x, train_y, opt_hyp, replace_old)
        expanded_train_x = self._expand(train_x, self.num_states + self.num_actions,
                                        self._junk_states + self._junk_actions)
        expanded_train_y = self._expand(train_y, self.num_states, self._junk_states)
        self._ssm.update_model(expanded_train_x, expanded_train_y, opt_hyp, replace_old)

    @staticmethod
    def _expand(x: Tensor, real_dimen: int, junk_dimen: int) -> Tensor:
        N = x.size(0)
        assert_shape(x, (N, real_dimen))

        expanded = torch.empty((N, real_dimen + junk_dimen), device=x.device, dtype=x.dtype)
        expanded[:, :real_dimen] = x

        random = torch.zeros((N, junk_dimen), device=x.device, dtype=x.dtype)
        random = random
        expanded[:, real_dimen:] = random

        return expanded

    def _update_model(self, x_train: Tensor, y_train: Tensor) -> None:
        pass

    def _train_model(self, x_train: Tensor, y_train: Tensor) -> None:
        pass

    def collect_metrics(self) -> Dict[str, Any]:
        return self._ssm.collect_metrics()

    @property
    def parametric(self) -> bool:
        return self._ssm.parametric
