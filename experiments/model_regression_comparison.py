"""Script to compare exact gp and mc dropout models on a simple regression task."""

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from safe_exploration.ssm_cem import McDropoutSSM
from safe_exploration.ssm_pytorch.gaussian_process import ZeroMeanWithGrad


class GP:
    def __init__(self, x_train, y_train):
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._model = ExactGPModel(x_train, y_train, self._likelihood)

    def predict(self, x: Tensor):
        self._likelihood.eval()
        self._model.eval()
        pred = self._model(x)
        return pred.mean.unsqueeze(1), pred.variance.unsqueeze(1)

    def train(self, xs: Tensor, ys: Tensor, iterations: int) -> None:
        self._likelihood.train()
        self._model.train()

        # self._model.parameters() includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam([{'params': self._model.parameters()}, ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        for i in range(iterations):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self._model(xs)
            # Calc loss and backprop gradients
            loss = -mll(output, ys).sum()
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print(f'{i}: {loss.item()}')


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMeanWithGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _plot(axes, x_train, y_train, x_test, preds):
    should_show = False
    if axes is None:
        axes = plt.axes()
        should_show = True

    plt.axis([-8, 8, -4, 4])

    # Plot the base sin function.
    xs = np.arange(-4, 4, 0.05)
    axes.plot(xs, np.sin(xs))

    # Plot the trainin data.
    axes.scatter(x_train.numpy(), y_train.numpy(), zorder=10, s=4)

    pred_mean, pred_var = preds
    pred_mean = pred_mean.squeeze(1).detach().numpy()
    pred_std = pred_var.squeeze(1).sqrt().detach().numpy()

    # Plot the mean line.
    axes.plot(x_test.detach().numpy(), pred_mean)

    # Plot the uncertainty.
    for i in range(1, 4):
        axes.fill_between(x_test, (pred_mean - i * pred_std).flat, (pred_mean + i * pred_std).flat, color="#dddddd",
                          alpha=1.0 / i)

    if should_show:
        plt.show()


class McDropoutSSMConfig:
    mc_dropout_num_samples = 100
    mc_dropout_predict_std = False
    device = None
    mc_dropout_reinitialize = True

    def __init__(self, hidden_features, num_training_iterations):
        self.mc_dropout_hidden_features = hidden_features
        self.mc_dropout_training_iterations = num_training_iterations


def run_mcdropout(axes, x_train, y_train, x_test, hidden_layer_size: int, training_iter: int):
    hidden_features = [hidden_layer_size, hidden_layer_size]
    mcdropout = McDropoutSSM(McDropoutSSMConfig(hidden_features, training_iter), state_dimen=1, action_dimen=0)

    mcdropout._train_model(x_train.unsqueeze(1), y_train)

    _plot(axes, x_train, y_train, x_test, mcdropout.predict_raw(x_test.unsqueeze(1)))


def run_gp(axes, x_train, y_train, x_test):
    gp = GP(x_train, y_train)
    gp.train(x_train, y_train, iterations=50)

    _plot(axes, x_train, y_train, x_test, gp.predict(x_test.unsqueeze(1)))


def main():
    x_train = torch.rand(20) * 8 - 4
    y_train = torch.sin(x_train) + 1e-1 * torch.randn_like(x_train)

    x_test = torch.linspace(-8, 8, 160)

    axes = None
    run_gp(axes, x_train, y_train, x_test)
    run_mcdropout(axes, x_train, y_train, x_test, hidden_layer_size=20, training_iter=10000)
    run_mcdropout(axes, x_train, y_train, x_test, hidden_layer_size=20, training_iter=5000)
    run_mcdropout(axes, x_train, y_train, x_test, hidden_layer_size=20, training_iter=5000)
    run_mcdropout(axes, x_train, y_train, x_test, hidden_layer_size=20, training_iter=5000)


if __name__ == '__main__':
    main()
