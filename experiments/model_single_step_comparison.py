"""Demonstrates the uncertainty estimates of gp vs mc dropout on single steps in the environment."""
import functools
import os
from typing import Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict
from numpy import ndarray

from experiments import sacred_helper
from safe_exploration.environments.environments import InvertedPendulum, Environment
from safe_exploration.gp_reachability_pytorch import onestep_reachability
from safe_exploration.ssm_cem.ssm_cem import CemSSM
from safe_exploration.ssm_cem.dropout_ssm_cem import McDropoutSSM
from safe_exploration.ssm_cem.gp_ssm_cem import GpCemSSM
from safe_exploration.utils import get_device
from safe_exploration.visualization import utils_visualization

ex = sacred_helper.get_experiment()


def _sample(env: Environment, action: ndarray) -> ndarray:
    # Set std to zero so we end up exactly at the origin
    initial = env.reset(mean=np.array([0., 0.]), std=np.array([0., 0.]))
    assert np.allclose(initial, np.array([0., 0.]))
    _, new_state, _, _, _ = env.step(action)
    return new_state


def _gather_samples(env: Environment) -> Tuple[ndarray, ndarray]:
    num_samples = 20
    actions = np.random.uniform(env.u_min, env.u_max, (num_samples,))
    # For now the initial state is always the origin.
    x_train = np.zeros((num_samples, env.n_s + env.n_u))
    x_train[:, 2] = actions

    y_train = np.empty((num_samples, env.n_s))
    for i in range(x_train.shape[0]):
        y_train[i] = _sample(env, actions[i])

    return x_train, y_train


def _get_color(i) -> str:
    return f'C{i}'


def _pretty_print_metrics(metrics: Dict[str, float]) -> str:
    result = ''
    for k, v in metrics.items():
        result += f'{k}: {v:.4f}; '
    return result


def _train_and_plot_for_ssm(conf, env: Environment, ssm: CemSSM, x_train, y_train, x_test, axes, linestyle='-'):
    device = get_device(conf)
    x_train = torch.tensor(x_train).to(device)
    y_train = torch.tensor(y_train).to(device)
    ssm.update_model(x_train, y_train, opt_hyp=True)

    states = torch.zeros((x_test.shape[0], env.n_s)).to(device)
    actions = torch.tensor(x_test).to(device)

    ps, qs, _ = onestep_reachability(states, ssm, actions, torch.tensor(env.l_mu).to(device),
                                     torch.tensor(env.l_sigm).to(device), verbose=0)

    for i in range(ps.size(0)):
        p = ps[i].unsqueeze(1).detach().cpu().numpy()
        q = qs[i].detach().cpu().numpy()
        utils_visualization.plot_ellipsoid_2D(p, q, axes, n_points=30, color=_get_color(i), linewidth=0.5,
                                              linestyle=linestyle)


def _run_experiment(env: Environment, x_train, y_train, x_test, save_to_file, conf):
    # Adjust the axes rectange to make room for the footnote text containing dropout probabilities.
    ax1 = plt.axes([0.125, 0.15, 0.885, 0.85])

    plt.tight_layout()

    ax1.set_xlabel('angular velocity')
    ax1.set_ylabel('angle from vertical')

    ax1.scatter([0], [0], color='black', s=3)
    ax1.scatter(y_train[:, 0], y_train[:, 1], color='grey', s=2)

    for i in range(x_test.shape[0]):
        y_test = _sample(env, x_test[i])
        ax1.scatter(y_test[0], y_test[1], color=_get_color(i), s=4)

    # mc_dropout = GalConcreteDropoutSSM(conf, env.n_s, env.n_u)
    mc_dropout = McDropoutSSM(conf, env.n_s, env.n_u)
    _train_and_plot_for_ssm(conf, env, mc_dropout, x_train, y_train, x_test, axes=ax1)

    gp = GpCemSSM(conf, env.n_s, env.n_u)
    _train_and_plot_for_ssm(conf, env, gp, x_train, y_train, x_test, axes=ax1, linestyle='--')

    plt.figtext(0, 0, _pretty_print_metrics(mc_dropout.collect_metrics()), wrap=True)

    if save_to_file:
        folder = 'results_single_step'
        if not os.path.isdir(folder):
            os.mkdir(folder)
        file_name = f'{folder}/{conf.name}.png'
        plt.savefig(file_name)
        plt.clf()
    else:
        plt.show()


@ex.capture
def _conf(_run, i, training_iter, lengthscale, fixed_p: Optional[float]):
    conf = EasyDict(_run.config)
    conf.mc_dropout_training_iterations = training_iter
    conf.mc_dropout_lengthscale = lengthscale

    if fixed_p is not None:
        conf.cem_ssm = 'mc_dropout'
        conf.mc_dropout_type = 'fixed'
        conf.mc_dropout_predict_std = False
    else:
        conf.cem_ssm = 'mc_dropout_gal'
        conf.mc_dropout_type = 'concrete'
        conf.mc_dropout_predict_std = True

    conf.name = f'ls={lengthscale:.6f}_p={fixed_p}_iter={training_iter}_{i}'

    return conf


@ex.automain
def single_step_compare_main(_run):
    torch.set_default_dtype(torch.double)

    env = InvertedPendulum(verbosity=0)

    x_train, y_train = _gather_samples(env)
    x_test = np.expand_dims(np.arange(env.u_min_norm, env.u_max_norm, 0.2), 1)

    save_to_file = True
    run = functools.partial(_run_experiment, env, x_train, y_train, x_test, save_to_file)

    run(_conf(i=0, training_iter=3000, lengthscale=1e-4, fixed_p=0.0001))
    run(_conf(i=0, training_iter=3000, lengthscale=1e-4, fixed_p=0.001))
    run(_conf(i=0, training_iter=3000, lengthscale=1e-4, fixed_p=0.01))
    run(_conf(i=0, training_iter=3000, lengthscale=1e-4, fixed_p=0.1))
    run(_conf(i=0, training_iter=3000, lengthscale=1e-4, fixed_p=0.2))
    run(_conf(i=0, training_iter=3000, lengthscale=1e-4, fixed_p=0.3))
    run(_conf(i=0, training_iter=3000, lengthscale=1e-4, fixed_p=0.4))
    run(_conf(i=0, training_iter=3000, lengthscale=1e-4, fixed_p=0.5))
