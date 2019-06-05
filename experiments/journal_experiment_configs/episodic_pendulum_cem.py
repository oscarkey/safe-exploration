# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""
from experiments.journal_experiment_configs.defaultconfig_exploration import DefaultConfigExploration
from .defaultconfig_episode import DefaultConfigEpisode
import numpy as np


class Config(DefaultConfigEpisode):
    """
    Pendulum in the episodic setting, using CEM MPC (rather than Casadi).

    Use the DefaultConfigExploration rather than DefaultConfigEpisode, because the former is set up for pendulum and the
    latter is set up for cart pole. This means we have to override a load of exploration config values.
    """
    verbose = 0

    n_ep = 8
    n_steps = 50
    n_steps_init = 8
    n_rollouts_init = 5  # 5
    n_scenarios = 6  # 10

    obs_frequency = 1  # Only take an observation every k-th time step (k = obs_frequency)

    # environment
    env_name = "InvertedPendulum"

    solver_type = "safempc_cem"

    lqr_wx_cost = np.diag([1., 2.])
    lqr_wu_cost = 25 * np.eye(1)

    lin_prior = True
    prior_model = dict()
    prior_m = .1
    prior_b = 0.0
    prior_model["m"] = prior_m
    prior_model["b"] = prior_b

    env_options = dict()
    init_std = np.array([.05, .05])
    env_options["init_std"] = init_std

    render = True
    visualize = True
    plot_ellipsoids = False
    plot_trajectory = False


    def __init__(self):
        """ """
        # self.cost = super._generate_cost()
        super(Config, self).__init__(__file__)
        # self.cost = super(Config, self)._generate_cost()
        self.cost = lambda p_0, u_0, p_all, q_all, k_ff_safe, k_fb_safe, sigma_safe: 0
        self.rl_immediate_cost = lambda state: 0
