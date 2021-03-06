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

    obs_frequency = 1  # Only take an observation every k-th time step (k = obs_frequency)

    # environment
    env_name = "LunarLander"

    solver_type = "safempc_cem"

    lqr_wx_cost = np.diag([1., 2., 2., 2.])
    lqr_wu_cost = 1 * np.eye(2)

    lin_prior = True
    prior_model = dict()
    prior_m = .1
    prior_b = 0.0
    prior_model["m"] = prior_m
    prior_model["b"] = prior_b

    env_options = dict()
    init_std = np.array([.05, .05])
    env_options["init_std"] = init_std

    init_std_initial_data = np.array([1., 1., 1., 1.])
    init_m_initial_data = np.array([0., 0., 0., 0.])

    plot_ellipsoids = False
    plot_trajectory = False


    def __init__(self):
        """ """
        # self.cost = super._generate_cost()
        super(Config, self).__init__(__file__)
        # self.cost = super(Config, self)._generate_cost()
        self.cost = lambda p_0, u_0, p_all, q_all, k_ff_safe, k_fb_safe, sigma_safe: 0
        self.rl_immediate_cost = lambda state: 0
