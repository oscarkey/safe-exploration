"""Demonstrates the uncertainty estimates of gp vs mc dropout on single steps in the environment."""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict
from numpy import ndarray

from experiments import sacred_helper
from safe_exploration.environments import InvertedPendulum, Environment
from safe_exploration.gp_reachability_pytorch import onestep_reachability
from safe_exploration.ssm_cem import GpCemSSM, CemSSM, McDropoutSSM
from visualization import utils_visualization

ex = sacred_helper.get_experiment()


def _gather_samples(env: Environment) -> Tuple[ndarray, ndarray]:
    num_samples = 20
    actions = np.random.uniform(env.u_min, env.u_max, (num_samples,))
    # For now the initial state is always the origin.
    x_train = np.zeros((num_samples, env.n_s + env.n_u))
    x_train[:, 2] = actions

    y_train = np.empty((num_samples, env.n_s))
    for i in range(x_train.shape[0]):
        # Set std to zero so we end up exactly at the origin
        initial = env.reset(mean=np.array([0., 0.]), std=np.array([0., 0.]))
        assert np.allclose(initial, np.array([0., 0.]))
        _, new_state, _, _ = env.step(actions[i])
        y_train[i] = new_state

    return x_train, y_train


def _run_experiment(env: Environment, ssm: CemSSM, x_train, y_train, x_test):
    ssm.update_model(torch.tensor(x_train), torch.tensor(y_train), opt_hyp=True)

    states = torch.zeros((x_test.shape[0], env.n_s))
    actions = torch.tensor(x_test)

    ps, qs, _ = onestep_reachability(states, ssm, actions, torch.tensor(env.l_mu), torch.tensor(env.l_sigm))

    axes = plt.axes()
    axes.scatter([0], [0], color='r', s=4)
    axes.scatter(y_train[:, 0], y_train[:, 1], s=2)

    for i in range(ps.size(0)):
        p = ps[i].unsqueeze(1).detach().cpu().numpy()
        q = qs[i].detach().cpu().numpy()
        utils_visualization.plot_ellipsoid_2D(p, q, axes, n_points=30)

    plt.show()


@ex.automain
def single_step_compare_main(_run):
    torch.set_default_dtype(torch.double)

    conf = EasyDict(_run.config)
    env = InvertedPendulum()

    x_train, y_train = _gather_samples(env)
    x_test = np.expand_dims(np.arange(env.u_min, env.u_max, 0.1), 1)

    mc_dropout = McDropoutSSM(conf, env.n_s, env.n_u)
    _run_experiment(env, mc_dropout, x_train, y_train, x_test)

    mc_dropout = GpCemSSM(conf, env.n_s, env.n_u)
    _run_experiment(env, mc_dropout, x_train, y_train, x_test)
