# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""

from .defaultconfig_episode import DefaultConfigEpisode


class Config(DefaultConfigEpisode):
    """
    Cartpole in the episodic setting, using CEM MPC (rather than Casadi).
    """

    # environment

    solver_type = "safempc_cem"

    # safempc
    beta_safety = 2.0
    n_safe = 5
    n_perf = 0
    r = 1

    # rl cost function
    cost = None
    ilqr_init = False


    def __init__(self):
        """ """
        # self.cost = super._generate_cost()
        super(Config, self).__init__(__file__)
        self.cost = super(Config, self)._generate_cost()
