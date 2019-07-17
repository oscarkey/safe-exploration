# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""
from typing import Dict, Any

import numpy as np

from .defaultconfig_episode import DefaultConfigEpisode


class Config(DefaultConfigEpisode):
    """
    Pendulum in the episodic setting, using CEM MPC (rather than Casadi).

    Use the DefaultConfigExploration rather than DefaultConfigEpisode, because the former is set up for pendulum and the
    latter is set up for cart pole. This means we have to override a load of exploration config values.
    """
    verbose = 0

    obs_frequency = 1  # Only take an observation every k-th time step (k = obs_frequency)

    # environment
    env_name = "InvertedPendulum"

    solver_type = "safempc_cem"

    lin_prior = True
    prior_model = dict()
    prior_m = .1
    prior_b = 0.0
    prior_model["m"] = prior_m
    prior_model["b"] = prior_b

    env_options = dict()

    plot_ellipsoids = False
    plot_trajectory = False

    def __init__(self):
        """ """
        # self.cost = super._generate_cost()
        super(Config, self).__init__(__file__)
        # self.cost = super(Config, self)._generate_cost()
        self.cost = lambda p_0, u_0, p_all, q_all, k_ff_safe, k_fb_safe, sigma_safe: 0
        self.rl_immediate_cost = lambda state: 0

    def add_sacred_config(self, config: Dict[str, Any]) -> None:
        # Call the superclass to actually add the config.
        super().add_sacred_config(config)

        # Configure properties which depend on the sacred config.
        num_angles = self.pendulum_dimensions - 1
        state_dimen = num_angles * 2
        self.lqr_wx_cost = np.diag([1.] * num_angles + [2.] * num_angles)
        self.lqr_wu_cost = 25 * np.eye(num_angles)

        velocity_std = 0.05
        self.init_std = np.array([.05, ] * state_dimen)
        self.env_options["init_std"] = self.init_std

        self.init_std_initial_data = np.array([.1, ] * state_dimen)
        self.init_m_initial_data = [0., ] * state_dimen
