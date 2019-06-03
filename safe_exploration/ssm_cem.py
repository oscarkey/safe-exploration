"""Contains state space models for use with CemSafeMPC. These should all using PyTorch."""
from abc import abstractmethod

import gpytorch
import torch
from torch import Tensor

from .ssm_pytorch import BatchKernel, MultiOutputGP, utilities
from .utils import assert_shape


class CemSSM:
    """State space model interface for use with CEM MPC.

    Unlike StateSpaceModel it uses Tensors rather than NumPy arrays. It also does not specify the methods required only
    for Casadi.
    """

    def __init__(self, num_states: int, num_actions: int):
        self.num_states = num_states
        self.num_actions = num_actions

    @abstractmethod
    def predict_with_jacobians(self, states: Tensor, actions: Tensor) -> (Tensor, Tensor, Tensor):
        """Predict the next states and uncertainties, along with the jacobian of the state means.

        Currently only supports a single state action pair.
        TODO: Add batch support.

        :param states (1 x n_s) tensor of states
        :param actions (1 x n_u) tensor of actions
        :return (mean of next states [1 x n_s], variance of next states [1 x n_s], jacobian of mean [1 x n_s x n_s])
        """
        pass

    @abstractmethod
    def predict_without_jacobians(self, states: Tensor, actions: Tensor) -> (Tensor, Tensor):
        """Predict the next states and uncertainties.

        Currently only supports a single state action pair.
        TODO: Add batch support.

        :param states (1 x n_s) tensor of states
        :param actions (1 x n_u) tensor of actions
        :return (mean of next states [1 x n_s], variance of next states [1 x n_s]
        """
        pass

    @abstractmethod
    def update_model(self, train_x: Tensor, train_y: Tensor, opt_hyp=False, replace_old=False) -> None:
        """Incorporate the given data into the model.

        :param train_x [N x (n_s + n_u)]
        :param train_y [N x n_s]
        :param replace_old If True, replace all existing training data with this new training data. Otherwise, merge it.
        """
        pass


class GpCemSSM(CemSSM):
    """A SSM using GPyTorch, for the CEM implementation of SafeMPC.

    Compared to GPyTorchSSM, this uses PyTorch tensors all the way through, rather than converting to Numpy. It also
    does not implement the linearization and differentation functions which are only require for Casadi.
    """

    def __init__(self, state_dimen: int, action_dimen: int):
        super().__init__(state_dimen, action_dimen)

        self._gp = MultiOutputGP(train_x=None, train_y=None,
                                 kernel=BatchKernel([gpytorch.kernels.RBFKernel()] * state_dimen),
                                 likelihood=gpytorch.likelihoods.GaussianLikelihood(batch_size=state_dimen))
        self._gp.eval()

    def predict_with_jacobians(self, states: Tensor, actions: Tensor) -> (Tensor, Tensor, Tensor):
        x = self._join_states_actions(states, actions)

        # We need the gradient to compute the jacobians.
        x.requires_grad = True

        pred = self._gp(x)
        jac_mean = utilities.compute_jacobian(pred.mean, x).squeeze()

        return pred.mean, pred.variance, jac_mean

    def predict_without_jacobians(self, states: Tensor, actions: Tensor) -> (Tensor, Tensor):
        x = self._join_states_actions(states, actions)
        pred = self._gp(x)
        return pred.mean, pred.variance

    def _join_states_actions(self, states: Tensor, actions: Tensor) -> Tensor:
        assert_shape(states, (1, self.num_states))
        assert_shape(actions, (1, self.num_actions))
        return torch.cat((states, actions), dim=1)

    def update_model(self, train_x: Tensor, train_y: Tensor, opt_hyp=False, replace_old=False) -> None:
        # TODO: select datapoints by highest variance.
        if replace_old:
            x_new = train_x
            y_new = train_y
        else:
            x_new = torch.cat((self._gp.train_inputs, train_x), dim=0)
            y_new = torch.cat((self._gp.train_targets, train_y), dim=0)

        if opt_hyp:
            raise NotImplementedError

        # Hack because set_train_data() does not work if there was no previous data.
        self._gp.train_inputs = []
        self._gp.train_targets = torch.zeros((0))

        self._gp.set_train_data(x_new, y_new, strict=False)
