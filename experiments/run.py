#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:03:14 2017

@author: tkoller
"""

import sys
from typing import Optional, Tuple

import sacred.arg_parser
import torch
from sacred.observers import MongoObserver, FileStorageObserver

from experiments import sacred_auth_details
from experiments.journal_experiment_configs.default_config import DefaultConfig
from safe_exploration.episode_runner import run_episodic
from safe_exploration.exploration_runner import run_exploration
from safe_exploration.uncertainty_propagation_runner import run_uncertainty_propagation
from safe_exploration.utils_config import load_config, create_env, create_solver
from utils_sacred import SacredAggregatedMetrics

ex = sacred.Experiment()

config_updates, _ = sacred.arg_parser.get_config_updates(sys.argv)

# Disable saving to mongo using "with save_to_db=False"
if ("save_to_db" not in config_updates) or config_updates["save_to_db"]:
    mongo_observer = MongoObserver.create(url=sacred_auth_details.db_url, db_name='safe-exploration')
    ex.observers.append(mongo_observer)
else:
    ex.observers.append(FileStorageObserver.create('safe_exploration_results'))


@ex.config
def base_config():
    save_to_db = True
    # File name of the base config file.
    scenario_file = None

    # -- Environment
    render = True
    visualize = True

    # Type of state space model to use, one of exact_gp, mc_dropout.
    cem_ssm = 'mc_dropout'

    # -- Episodic
    # The number of repeats of the experiment, over which we will average the metrics.
    n_scenarios = 6
    # Number of episodes in each repeat. Each episode lasts until there is a safety failure, up to n_steps
    n_ep = 8
    # Maximum number of steps in a single episode.
    n_steps = 50

    # -- CemSafeMPC
    mpc_time_horizon = 2
    cem_num_rollouts = 20
    cem_num_elites = 3
    cem_num_iterations = 8
    # Whether to plot the ellipsoids generated during the CEM optimisation process.
    plot_cem_optimisation = False

    # -- MC dropout SSM
    mc_dropout_training_iterations = 1000
    # List giving number of units in each hidden layer.
    mc_dropout_hidden_features = [200, 200]
    # Number of times we will sample the network during the forward pass, to compute the mean + var.
    mc_dropout_num_samples = 200

    # -- Exact GP SSM
    exact_gp_training_iterations = 200


def check_config_conflicts(conf: DefaultConfig) -> Tuple[bool, str]:
    """ Check if there are conflicting options in the Config

    Parameters
    ----------
    conf: Config
        The config file

    Returns
    -------
    has_conflict: Bool
        True, if there is a conflict in the config
    conflict_str: String
        The error message

    """

    has_conflict = False
    conflict_str = ""
    if conf.task == "exploration" and not conf.solver_type in ("safempc", "safempc_cem"):
        return True, "Exploration task only allowed with safempc solver"
    elif conf.task == "uncertainty_propagation" and not conf.solver_type == "safempc":
        return True, "Uncertainty propagation task only allowed with safempc solver"

    return has_conflict, conflict_str


@ex.capture
def _run_scenario(_run, scenario_file: Optional[str]):
    """ Run the specified scenario

    Parameters
    ----------
    args:
        The parsed arguments (see create_parser for details)
    """
    if scenario_file is None:
        raise ValueError('Must provide a scenario file!')

    conf = load_config(scenario_file)

    conf.add_sacred_config(_run.config)

    conflict, conflict_str = check_config_conflicts(conf)
    if conflict:
        raise ValueError("There are conflicting settings: {}".format(conflict_str))

    metrics = SacredAggregatedMetrics(_run)

    env = create_env(conf.env_name, conf.env_options)

    solver, safe_policy = create_solver(conf, env)

    task = conf.task
    if task == "exploration":
        run_exploration(conf, conf.visualize)
    elif task == "episode_setting":
        run_episodic(conf, metrics)

    elif task == "uncertainty_propagation":
        run_uncertainty_propagation(env, solver, conf)


@ex.automain
def main():
    torch.set_default_dtype(torch.double)
    _run_scenario()
