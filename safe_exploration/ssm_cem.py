"""Contains state space models for use with CemSafeMPC. These should all using PyTorch."""
from abc import abstractmethod, ABC
from typing import Tuple, Optional

import bnn
import gpytorch
import matplotlib.pyplot as plt
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


class GpCemSSM(CemSSM):
    """A SSM using an exact GP from GPyTorch, for the CEM implementation of SafeMPC.

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
        pred = self._model(z)
        return pred.mean, pred.variance

    def _update_model(self, x_train: Tensor, y_train: Tensor) -> None:
        # Hack because set_train_data() does not work if previous data was None.
        self._model.train_inputs = []
        self._model.train_targets = torch.zeros((0))

        self._model.set_train_data(x_train, y_train.transpose(0, 1), strict=False)

    def _train_model(self, x_train: Tensor, y_train: Tensor) -> None:
        self._model.train()
        self._likelihood.train()

        # self._model.parameters() includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam([{'params': self._model.parameters()}, ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        training_iter = 200
        print(f'Training GP on {self.x_train.size(0)} data points for {training_iter} iterations...')
        losses = []
        for i in range(training_iter):
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


class McDropoutSSM(CemSSM):
    """BNN, approximated using MC dropout, state space model.

    Uses the "bnn" package from https://github.com/anassinator/bnn
    """

    def __init__(self, state_dimen: int, action_dimen: int):
        super().__init__(state_dimen, action_dimen)

        self._num_particles = 100

        in_features = state_dimen + action_dimen
        out_features = state_dimen
        self._model = bnn.bayesian_model(in_features, out_features, hidden_features=[200, 200])
        self._optimizer = torch.optim.Adam(p for p in self._model.parameters() if p.requires_grad)

    def predict_with_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z = self._join_states_actions(states, actions)
        pred_mean, pred_var = self.predict_raw(z)

        def mean_func(x: Tensor):
            return self.predict_raw(x)[0]

        pred_mean_jac = utilities.compute_jacobian_fast(mean_func, z, num_outputs=self.num_states)

        return pred_mean, pred_var, pred_mean_jac

    def predict_without_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        # To get the variance, we sample _num_particles times from the network.
        z = self._join_states_actions(states, actions)
        return self.predict_raw(z)

    def predict_raw(self, z: Tensor):
        N = z.size(0)
        assert_shape(z, (N, self.num_states + self.num_actions))

        z_particles = z.repeat((self._num_particles, 1, 1))
        output = self._model(z_particles)

        # TODO: Do we need to predict the variance too? See:
        # https://github.com/anassinator/bnn/blob/master/examples/sin_x.ipynb

        pred_mean = output.mean(dim=0)
        pred_var = output.var(dim=0)

        return pred_mean, pred_var

    def _update_model(self, x_train: Tensor, y_train: Tensor) -> None:
        # Nothing to do. We do not store the training data, just incorporate it in the model in _train_model().
        pass

    def _train_model(self, x_train: Tensor, y_train: Tensor) -> None:
        training_iter = 1000
        print(f'Training BNN on {self.x_train.size(0)} data points for {training_iter} iterations...')
        losses = []
        for i in range(training_iter):
            self._optimizer.zero_grad()
            output = self._model(x_train, resample=True)
            loss = (-self._gaussian_log_likelihood(y_train, output) + 1e-2 * self._model.regularization()).mean()
            loss.backward()
            losses.append(loss.item())
            self._optimizer.step()
        print(f'Training complete. Final losses: {losses[-4]:.2f} {losses[-3]:.2f} {losses[-2]:.2f} {losses[-1]:.2f}')

    @staticmethod
    def _gaussian_log_likelihood(targets, pred_means, pred_stds=None):
        """Taken from https://github.com/anassinator/bnn/blob/master/examples/sin_x.ipynb"""
        deltas = pred_means - targets
        return -(deltas ** 2).sum(-1) * 0.5


def test_plot(ssm: CemSSM, train_x: Tensor, train_y: Tensor):
    states = torch.range(0., 5., 0.1).unsqueeze(1)
    actions = torch.empty((states.size(0), 0))
    mu, std = ssm.predict_without_jacobians(states, actions)

    xs = states.squeeze(1).detach().numpy()
    mu = mu.detach().numpy()
    std = std.detach().numpy()

    plt.plot(xs, mu)

    for i in range(1, 4):
        plt.gca().fill_between(xs.flat, (mu - i * std).flat, (mu + i * std).flat, color="#dddddd", alpha=1.0 / i,
                               label="Confidence")

    plt.scatter(train_x.squeeze(1).numpy(), train_y.squeeze(1).numpy())
    plt.show()
