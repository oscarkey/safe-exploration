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
    scenario_file = None
    # Type of state space model to use, one of exact_gp, mc_dropout.
    cem_ssm = 'exact_gp'
    # Whether to plot the ellipsoids generated during the CEM optimisation process.
    plot_cem_optimisation = True


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

    env = create_env(conf.env_name, conf.env_options)

    solver, safe_policy = create_solver(conf, env)

    task = conf.task
    if task == "exploration":
        run_exploration(conf, conf.visualize)
    elif task == "episode_setting":
        run_episodic(conf)

    elif task == "uncertainty_propagation":
        run_uncertainty_propagation(env, solver, conf)


@ex.automain
def main():
    torch.set_default_dtype(torch.double)
    _run_scenario()
