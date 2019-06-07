# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:09:14 2017

@author: tkoller
"""
from .defaultconfig_exploration import DefaultConfigExploration


class Config(DefaultConfigExploration):
    """
    Options class for the exploration setting
    """
    verbose = 0
    static_exploration = False

    solver_type = "safempc_cem"

    # safempc
    beta_safety = 2.0
    n_safe = 2
    n_perf = 0
    r = 1

    def __init__(self):
        """ """
        super(Config, self).__init__(__file__)