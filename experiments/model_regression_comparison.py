"""Script to compare exact gp and mc dropout models on a simple regression task."""
import functools
import os
from typing import Optional

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict
from torch import Tensor

from experiments import sacred_helper
from safe_exploration.ssm_cem.gal_concrete_dropout import GalConcreteDropoutSSM
from safe_exploration.ssm_cem.ssm_cem import McDropoutSSM
from safe_exploration.ssm_pytorch.gaussian_process import ZeroMeanWithGrad
from safe_exploration.utils import get_device

ex = sacred_helper.get_experiment()


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


def _plot(x_train, y_train, x_test, preds, file_name: Optional[str] = None):
    x_train = x_train.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    x_test = x_test.detach().cpu().numpy()

    axes = plt.axes()

    plt.axis([-8, 8, -4, 4])

    # Plot the base sin function.
    xs = np.arange(-4, 4, 0.05)
    axes.plot(xs, np.sin(xs))

    # Plot the trainin data.
    axes.scatter(x_train, y_train, zorder=10, s=4)

    pred_mean, pred_var = preds
    pred_mean = pred_mean.squeeze(1).detach().cpu().numpy()
    pred_std = pred_var.squeeze(1).sqrt().detach().cpu().numpy()

    # Plot the mean line.
    axes.plot(x_test, pred_mean)

    # Plot the uncertainty.
    for i in range(1, 4):
        axes.fill_between(x_test, (pred_mean - i * pred_std).flat, (pred_mean + i * pred_std).flat, color="#dddddd",
                          alpha=1.0 / i)

    if file_name is not None:
        print(f'Saved fig to {file_name}')
        plt.savefig(file_name)
        plt.clf()
    else:
        plt.show()


def _run_mcdropout(conf, x_train, y_train, x_test):
    if conf.impl == 'lib':
        mcdropout = McDropoutSSM(conf, state_dimen=1, action_dimen=0)
    elif conf.impl == 'gal':
        mcdropout = GalConcreteDropoutSSM(conf, state_dimen=1, action_dimen=0)
    else:
        raise ValueError(f'Unknown impl {conf.impl}')

    mcdropout._train_model(x_train.unsqueeze(1), y_train)

    folder = 'results_regression'
    if not os.path.isdir(folder):
        os.mkdir(folder)
    file_name = f'{folder}/{conf.name}.png'
    # file_name = None
    _plot(x_train, y_train, x_test, mcdropout.predict_raw(x_test.unsqueeze(1)), file_name)
    print(f'final p \'{conf.name}\': {mcdropout.get_dropout_probabilities()}')


def _run_gp(x_train, y_train, x_test):
    gp = GP(x_train, y_train)
    gp.train(x_train, y_train, iterations=50)

    _plot(x_train, y_train, x_test, gp.predict(x_test.unsqueeze(1)))


@ex.capture
def _conf(_run, i: int, impl: str, hidden_layer_size: int, training_iter: int, dropout_type: str, dropout_p: float):
    conf = EasyDict(_run.config)
    conf.mc_dropout_predict_std = True
    conf.mc_dropout_reinitialize = True
    conf.mc_dropout_hidden_features = [hidden_layer_size, hidden_layer_size]
    conf.mc_dropout_training_iterations = training_iter
    conf.mc_dropout_type = dropout_type
    conf.mc_dropout_concrete_initial_probability = dropout_p
    conf.mc_dropout_fixed_probability = dropout_p

    conf.impl = impl
    if impl == 'gal':
        assert dropout_type == 'concrete'
        assert dropout_p == 0.1

    conf.name = f'{impl}_{dropout_type}_iter={training_iter}_hiddensize={hidden_layer_size}_p={dropout_p:.3f}_{i}'

    return conf


@ex.automain
def regression_comparison_main(_run):
    conf = EasyDict(_run.config)

    x_train = torch.rand(20) * 8 - 4
    y_train = torch.sin(x_train) + 1e-1 * torch.randn_like(x_train)
    x_test = torch.linspace(-8, 8, 160)

    x_train = x_train.to(get_device(conf))
    y_train = y_train.to(get_device(conf))
    x_test = x_test.to(get_device(conf))

    run = functools.partial(_run_mcdropout, x_train=x_train, y_train=y_train, x_test=x_test)

    # Search over training iterations and repeats for gal implementation.
    run(_conf(i=0, impl='gal', hidden_layer_size=20, training_iter=1, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=1, impl='gal', hidden_layer_size=20, training_iter=1, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=2, impl='gal', hidden_layer_size=20, training_iter=3000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=0, impl='gal', hidden_layer_size=20, training_iter=5000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=1, impl='gal', hidden_layer_size=20, training_iter=5000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=2, impl='gal', hidden_layer_size=20, training_iter=5000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=0, impl='gal', hidden_layer_size=20, training_iter=7000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=1, impl='gal', hidden_layer_size=20, training_iter=7000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=2, impl='gal', hidden_layer_size=20, training_iter=7000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=0, impl='gal', hidden_layer_size=20, training_iter=9000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=1, impl='gal', hidden_layer_size=20, training_iter=9000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=2, impl='gal', hidden_layer_size=20, training_iter=9000, dropout_type='concrete', dropout_p=0.1))

    # Search over training iterations and repeats.
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=3000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=1, impl='lib', hidden_layer_size=20, training_iter=3000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=2, impl='lib', hidden_layer_size=20, training_iter=3000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=5000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=1, impl='lib', hidden_layer_size=20, training_iter=5000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=2, impl='lib', hidden_layer_size=20, training_iter=5000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=7000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=1, impl='lib', hidden_layer_size=20, training_iter=7000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=2, impl='lib', hidden_layer_size=20, training_iter=7000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=1, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='concrete', dropout_p=0.1))
    run(_conf(i=2, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='concrete', dropout_p=0.1))

    # Search over training iterations and repeats for fixed dropout probabilities.
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=3000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=1, impl='lib', hidden_layer_size=20, training_iter=3000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=2, impl='lib', hidden_layer_size=20, training_iter=3000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=5000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=1, impl='lib', hidden_layer_size=20, training_iter=5000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=2, impl='lib', hidden_layer_size=20, training_iter=5000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=7000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=1, impl='lib', hidden_layer_size=20, training_iter=7000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=2, impl='lib', hidden_layer_size=20, training_iter=7000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=1, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=2, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='fixed', dropout_p=0.1))

    # Search over different dropout probabilities.
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='fixed', dropout_p=0.05))
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='fixed', dropout_p=0.1))
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='fixed', dropout_p=0.2))
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='fixed', dropout_p=0.3))
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='fixed', dropout_p=0.4))
    run(_conf(i=0, impl='lib', hidden_layer_size=20, training_iter=9000, dropout_type='fixed', dropout_p=0.5))

    # run_gp(axes, x_train, y_train, x_test)
