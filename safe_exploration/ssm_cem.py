"""Contains state space models for use with CemSafeMPC. These should all using PyTorch."""
from abc import abstractmethod, ABC
from typing import Tuple, Optional

import gpytorch
import torch
from torch import Tensor

from .ssm_pytorch import BatchKernel, MultiOutputGP, utilities
from .utils import assert_shape


class CemSSM(ABC):
    """State space model interface for use with CEM MPC.

    Unlike StateSpaceModel it uses Tensors rather than NumPy arrays. It also does not specify the methods required only
    for Casadi.
    """

    def __init__(self, num_states: int, num_actions: int):
        self.num_states = num_states
        self.num_actions = num_actions

    @property
    @abstractmethod
    def x_train(self) -> Optional[Tensor]:
        pass

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
    def predict_raw(self, z: Tensor):
        """Predict using stacked state and action.

        :param z: [1 x (n_s + n_u)]
        :returns: [1 x n_s], [1 x n_s]
        """
        pass

    @abstractmethod
    def update_model(self, train_x: Tensor, train_y: Tensor, opt_hyp=False, replace_old=False) -> None:
        """Incorporate the given data into the model.

        :param train_x: [N x (n_s + n_u)]
        :param train_y: [N x n_s]
        :param replace_old: If True, replace all existing training data with this new training data. Otherwise, merge
                            it.
        """
        pass


class GpCemSSM(CemSSM):
    """A SSM using GPyTorch, for the CEM implementation of SafeMPC.

    Compared to GPyTorchSSM, this uses PyTorch tensors all the way through, rather than converting to Numpy. It also
    does not implement the linearization and differentation functions which are only require for Casadi.
    """

    def __init__(self, state_dimen: int, action_dimen: int, model: Optional[MultiOutputGP] = None):
        """Constructs a new instance.

        :param model: Set during unit testing to inject a model.
        """
        super().__init__(state_dimen, action_dimen)

        self._likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=state_dimen)
        if model is None:
            self._model = MultiOutputGP(train_x=None, train_y=None,
                                        kernel=BatchKernel([gpytorch.kernels.RBFKernel()] * state_dimen),
                                        likelihood=self._likelihood)
        else:
            self._model = model
        self._model.eval()

    @property
    def x_train(self) -> Optional[Tensor]:
        """Returns the x values of the current training data."""
        if self._model.train_inputs is None:
            return None
        else:
            # train_inputs is list of [output dimen x n x state dimen + action dimen]
            # It is repeated over output dimen, so we can ignore this.
            # We want to concatenate over the list in the batch dimension, n.
            return torch.cat([x[0, :, :] for x in self._model.train_inputs], dim=0)

    @property
    def y_train(self) -> Optional[Tensor]:
        if self._model.train_targets is None:
            return None
        else:
            return self._model.train_targets.transpose(0, 1)

    def predict_with_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z = self._join_states_actions(states, actions)

        pred = self._model(z)
        pred_mean = pred.mean.transpose(0, 1)
        pred_var = pred.variance.transpose(0, 1)

        # TODO: avoid a second forward pass through the model.
        def mean_func(x: Tensor):
            pred1 = self._model(x)
            res = pred1.mean.transpose(0, 1)
            return res

        pred_mean_jac = utilities.compute_jacobian_fast(mean_func, z, num_outputs=self.num_states)

        N = states.size(0)
        assert_shape(pred_mean, (N, self.num_states))
        assert_shape(pred_var, (N, self.num_states))
        assert_shape(pred_mean_jac, (N, self.num_states, self.num_states + self.num_actions))

        return pred_mean, pred_var, pred_mean_jac

    def predict_without_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        z = self._join_states_actions(states, actions)

        pred = self._model(z)
        pred_mean = pred.mean.transpose(0, 1)
        pred_var = pred.variance.transpose(0, 1)

        N = states.size(0)
        assert_shape(pred_mean, (N, self.num_states))
        assert_shape(pred_var, (N, self.num_states))

        return pred_mean, pred_var

    def predict_raw(self, z: Tensor):
        N = z.size(0)
        assert_shape(z, (N, self.num_states + self.num_actions))
        pred = self._model(z)
        return pred.mean, pred.variance

    def _join_states_actions(self, states: Tensor, actions: Tensor) -> Tensor:
        N = states.size(0)
        assert_shape(states, (N, self.num_states))
        assert_shape(actions, (N, self.num_actions))
        return torch.cat((states, actions), dim=1)

    def update_model(self, train_x: Tensor, train_y: Tensor, opt_hyp=False, replace_old=False) -> None:
        assert train_x.size(0) == train_y.size(0), 'Batch dimensions must be equal.'
        # TODO: select datapoints by highest variance.
        if replace_old or self._model.train_inputs is None or self._model.train_targets is None:
            x_new = train_x
            y_new = train_y
        else:
            x_new = torch.cat((self.x_train, train_x), dim=0)
            y_new = torch.cat((self.y_train, train_y), dim=0)

        # Hack because set_train_data() does not work if previous data was None.
        self._model.train_inputs = []
        self._model.train_targets = torch.zeros((0))

        self._model.set_train_data(x_new, y_new.transpose(0, 1), strict=False)

        if opt_hyp:
            self._run_optimisation(x_new, y_new)

    def _run_optimisation(self, train_x: Tensor, train_y: Tensor):
        self._model.train()
        self._likelihood.train()

        # self._model.parameters() includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam([{'params': self._model.parameters()}, ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        training_iter = 200
        print(f'Training GP for {training_iter} iterations...')
        losses = []
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self._model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y.transpose(0, 1)).sum()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print(f'Training complete. Final losses: {losses[-4]:.2f} {losses[-3]:.2f} {losses[-2]:.2f} {losses[-1]:.2f}')

        self._model.eval()
        self._likelihood.eval()
