"""Script to compare exact gp and mc dropout models on a simple regression task."""
import functools
import os
from typing import List

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict

from experiments import sacred_helper
from safe_exploration.ssm_cem.gal_concrete_dropout import GalConcreteDropoutSSM
from safe_exploration.ssm_cem.gp_ssm_cem import GpCemSSM
from safe_exploration.ssm_cem.ssm_cem import McDropoutSSM, CemSSM
from safe_exploration.ssm_pytorch.gaussian_process import ZeroMeanWithGrad
from safe_exploration.utils import get_device

ex = sacred_helper.get_experiment()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMeanWithGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _plot(axes, x_train, y_train, x_test, preds, text: str):
    x_train = x_train.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    x_test = x_test.detach().cpu().numpy()

    plt.axis([-12, 12, -4, 4])

    # Plot the base sin function.
    xs = np.arange(-12, 12, 0.05)
    axes.plot(xs, _true_func(xs), color='C0', linestyle='--')

    # Plot the trainin data.
    axes.scatter(x_train, y_train, zorder=10, s=4, color='C0')

    pred_mean, pred_var = preds
    pred_mean = pred_mean.squeeze().detach().cpu().numpy()
    pred_std = pred_var.squeeze().sqrt().detach().cpu().numpy()

    # Plot the mean line.
    axes.plot(x_test, pred_mean, color='C1')

    axes.text(0, 0, text, transform=axes.transAxes, wrap=True)

    # Plot the uncertainty.
    for i in range(1, 4):
        axes.fill_between(x_test, (pred_mean - i * pred_std).flat, (pred_mean + i * pred_std).flat, color="#dddddd",
                          alpha=1.0 / i)


def _get_ssm(conf) -> CemSSM:
    if conf.cem_ssm == 'exact_gp':
        return GpCemSSM(conf, state_dimen=1, action_dimen=0)
    elif conf.cem_ssm == 'mc_dropout':
        return McDropoutSSM(conf, state_dimen=1, action_dimen=0)
    elif conf.cem_ssm == 'mc_dropout_gal':
        return GalConcreteDropoutSSM(conf, state_dimen=1, action_dimen=0)
    else:
        raise ValueError(f'Unknown ssm {conf.ssm_type}')


def _run_experiment(conf, x_train, y_train, x_test, save_to_file):
    ssm = _get_ssm(conf)

    ssm.update_model(x_train.unsqueeze(1), y_train.unsqueeze(1), opt_hyp=True)

    preds = ssm.predict_raw(x_test.unsqueeze(1))

    axes = plt.axes()
    # plt.tight_layout()
    # plt.plot(xs, ys, color='C7')
    _plot(axes, x_train, y_train, x_test, preds, text="")

    if save_to_file:
        folder = 'results_regression'
        if not os.path.isdir(folder):
            os.mkdir(folder)
        file_name = f'{folder}/{conf.name}.png'
        plt.savefig(file_name)
        plt.clf()
        print(f'Saved fig to {file_name}')
    else:
        plt.show()


@ex.capture
def _dropout_conf(_run, i: int, impl: str, hidden_layer_size: int, training_iter: int, dropout_type: str = 'concrete',
                  dropout_p: float = 0.1, input_dropout=True, length_scale=1e-4):
    conf = EasyDict(_run.config)
    conf.mc_dropout_predict_std = True
    conf.mc_dropout_reinitialize = True
    conf.mc_dropout_hidden_features = [hidden_layer_size, hidden_layer_size]
    conf.mc_dropout_training_iterations = training_iter
    conf.mc_dropout_type = dropout_type
    conf.mc_dropout_concrete_initial_probability = dropout_p
    conf.mc_dropout_fixed_probability = dropout_p
    conf.mc_dropout_on_input = input_dropout
    conf.mc_dropout_lengthscale = length_scale

    conf.name = f'{impl}_{dropout_type}_ls={length_scale:.5f}' \
        f'_iter={training_iter}_indrop={input_dropout}_hidden={hidden_layer_size}_p={dropout_p:.3f}_{i}'

    return conf


@ex.capture
def _gp_conf(_run, i, kernel, nn_kernel_layers: List[int]):
    conf = EasyDict(_run.config)
    conf.cem_ssm = 'exact_gp'
    conf.exact_gp_kernel = kernel
    conf.exact_gp_training_iterations = 1000
    conf.nn_kernel_layers = nn_kernel_layers

    layers_name = str(nn_kernel_layers).replace(' ', '')
    conf.name = f'{kernel}_{layers_name}_{i}'

    return conf


def _true_func(x):
    return np.sin(x - 0.8)  # return 0.2 * x + .5


@ex.automain
def regression_comparison_main(_run):
    torch.set_default_dtype(torch.double)

    conf = EasyDict(_run.config)

    x_train = np.concatenate([np.arange(-2, -0, 0.2), np.arange(3, 4, 0.2)])
    y_train = _true_func(x_train)
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)

    x_test = torch.linspace(-12, 12, 160)

    x_train = x_train.to(get_device(conf))
    y_train = y_train.to(get_device(conf))
    x_test = x_test.to(get_device(conf))

    run = functools.partial(_run_experiment, x_train=x_train, y_train=y_train, x_test=x_test, save_to_file=True)

    run(_gp_conf(i=0, kernel='nn', nn_kernel_layers=[16, 16, 1]))
    run(_gp_conf(i=1, kernel='nn', nn_kernel_layers=[16, 16, 1]))
    run(_gp_conf(i=2, kernel='nn', nn_kernel_layers=[16, 16, 1]))
    run(_gp_conf(i=0, kernel='nn', nn_kernel_layers=[16, 16, 32]))
    run(_gp_conf(i=1, kernel='nn', nn_kernel_layers=[16, 16, 32]))
    run(_gp_conf(i=2, kernel='nn', nn_kernel_layers=[16, 16, 32]))
    run(_gp_conf(i=0, kernel='nn', nn_kernel_layers=[16, 32, 64]))
    run(_gp_conf(i=1, kernel='nn', nn_kernel_layers=[16, 32, 64]))
    run(_gp_conf(i=2, kernel='nn', nn_kernel_layers=[16, 32, 64]))
    run(_gp_conf(i=0, kernel='nn', nn_kernel_layers=[16, 32, 128]))
    run(_gp_conf(i=1, kernel='nn', nn_kernel_layers=[16, 32, 128]))
    run(_gp_conf(i=2, kernel='nn', nn_kernel_layers=[16, 32, 128]))
    run(_gp_conf(i=0, kernel='nn', nn_kernel_layers=[32, 64, 256]))
    run(_gp_conf(i=1, kernel='nn', nn_kernel_layers=[32, 64, 256]))
    run(_gp_conf(i=2, kernel='nn', nn_kernel_layers=[32, 64, 256]))
    run(_gp_conf(i=0, kernel='nn', nn_kernel_layers=[32, 64, 128, 256]))
    run(_gp_conf(i=1, kernel='nn', nn_kernel_layers=[32, 64, 128, 256]))
    run(_gp_conf(i=2, kernel='nn', nn_kernel_layers=[32, 64, 128, 256]))
    run(_gp_conf(i=0, kernel='nn', nn_kernel_layers=[32, 64, 256, 512]))
    run(_gp_conf(i=1, kernel='nn', nn_kernel_layers=[32, 64, 256, 512]))
    run(_gp_conf(i=2, kernel='nn', nn_kernel_layers=[32, 64, 256, 512]))
    run(_gp_conf(i=0, kernel='nn', nn_kernel_layers=[64, 128, 256, 512]))
    run(_gp_conf(i=1, kernel='nn', nn_kernel_layers=[64, 128, 256, 512]))
    run(_gp_conf(i=2, kernel='nn', nn_kernel_layers=[64, 128, 256, 512]))
    run(_gp_conf(i=0, kernel='nn', nn_kernel_layers=[32, 64, 256, 512, 1024]))
    run(_gp_conf(i=1, kernel='nn', nn_kernel_layers=[32, 64, 256, 512, 1024]))
    run(_gp_conf(i=2, kernel='nn', nn_kernel_layers=[32, 64, 256, 512, 1024]))

    return
