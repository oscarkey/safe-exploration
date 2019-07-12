"""State space models for CEM MPC using gaussian processes."""
from typing import Optional, Tuple, Dict

import gpytorch
import torch
from gpytorch.kernels import ScaleKernel, RBFKernel
from torch import Tensor

from .ssm_cem import CemSSM
from ..ssm_pytorch import MultiOutputGP, utilities
from ..utils import get_device, assert_shape


class GpCemSSM(CemSSM):
    """A SSM using an exact GP from GPyTorch, for the CEM implementation of SafeMPC.

    Compared to GPyTorchSSM, this uses PyTorch tensors all the way through, rather than converting to Numpy. It also
    does not implement the linearization and differentation functions which are only require for Casadi.
    """

    def __init__(self, conf, state_dimen: int, action_dimen: int, model: Optional[MultiOutputGP] = None):
        """Constructs a new instance.

        :param model: Set during unit testing to inject a model.
        """
        super().__init__(state_dimen, action_dimen)

        self._training_iterations = conf.exact_gp_training_iterations

        self._likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=state_dimen)
        if model is None:
            kernel = ScaleKernel(RBFKernel(batch_size=state_dimen), batch_size=state_dimen)
            self._model = MultiOutputGP(train_x=None, train_y=None, likelihood=self._likelihood, kernel=kernel,
                                        num_outputs=state_dimen)
        else:
            self._model = model
        self._model = self._model.to(get_device(conf))
        self._model.eval()

    def predict_with_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z = self._join_states_actions(states, actions)

        pred_mean, pred_var = self._predict(z)

        # TODO: avoid a second forward pass through the model.
        def mean_func(x: Tensor):
            return self._predict(x)[0]

        pred_mean_jac = utilities.compute_jacobian_fast(mean_func, z, num_outputs=self.num_states)

        N = states.size(0)
        assert_shape(pred_mean_jac, (N, self.num_states, self.num_states + self.num_actions))

        return pred_mean, pred_var, pred_mean_jac

    def predict_without_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        z = self._join_states_actions(states, actions)
        return self._predict(z)

    def _predict(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        pred_mean, pred_var = self.predict_raw(z)
        pred_mean = pred_mean.transpose(0, 1)
        pred_var = pred_var.transpose(0, 1)

        N = z.size(0)
        assert_shape(pred_mean, (N, self.num_states))
        assert_shape(pred_var, (N, self.num_states))

        return pred_mean, pred_var

    def predict_raw(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        N = z.size(0)
        assert_shape(z, (N, self.num_states + self.num_actions))
        pred = self._likelihood(self._model(z))
        return pred.mean, pred.variance

    def _update_model(self, x_train: Tensor, y_train: Tensor) -> None:
        # Hack because set_train_data() does not work if previous data was None.
        self._model.train_inputs = []
        self._model.train_targets = torch.zeros((0), device=x_train.device)

        self._model.set_train_data(x_train, y_train.transpose(0, 1), strict=False)

    def _train_model(self, x_train: Tensor, y_train: Tensor) -> None:
        self._model.train()
        self._likelihood.train()

        # self._model.parameters() includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam([{'params': self._model.parameters()}, ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        print(f'Training GP on {self.x_train.size(0)} data points for {self._training_iterations} iterations...')
        losses = []
        for i in range(self._training_iterations):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self._model(x_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train.transpose(0, 1)).sum()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print(f'Training complete. Final losses: {losses[-4]:.2f} {losses[-3]:.2f} {losses[-2]:.2f} {losses[-1]:.2f}')

        self._model.eval()
        self._likelihood.eval()

    def collect_metrics(self) -> Dict[str, float]:
        # Currently we don't have any metrics.
        return {}
