"""Sets up Sacred, and contains the default parameters."""
import sys

import sacred
from sacred.observers import MongoObserver, FileStorageObserver

from experiments import sacred_auth_details

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

    # Device to force Tensors to. None, or a valid pytorch device (e.g. cpu, cuda:0)
    device = None

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
    # If True then constrain all the states, otherwise just the terminal state.
    use_state_constraint = True

    # -- MC dropout SSM
    mc_dropout_code_version = 2
    mc_dropout_training_iterations = 1000
    # List giving number of units in each hidden layer.
    mc_dropout_hidden_features = [64, 64]
    # Number of times we will sample the network during the forward pass, to compute the mean + var.
    mc_dropout_num_samples = 40
    # Whether to predict the aleatoric uncertainty as well as computing the epistemic uncertainty. (?)
    mc_dropout_predict_std = True
    # Whether to reinitialize the network weights before training.
    mc_dropout_reinitialize = True
    # Whether to use a fixed dropout probability or concrete dropout. One of 'fixed' or 'concrete'.
    mc_dropout_type = 'concrete'
    # The initial dropout rate, if mc_dropout_type = 'concrete'.
    mc_dropout_concrete_initial_probability = 0.5
    # The dropout rate, if mc_dropout_type = 'fixed'.
    mc_dropout_fixed_probability = 0.1

    # -- Exact GP SSM
    exact_gp_code_version = 2
    exact_gp_training_iterations = 200


@ex.named_config
def server_config():
    """Contains appropriate flags for when running on a headless gpu server."""
    render = False
    visualize = False


def get_experiment():
    """Returns the Sacred experiment object."""
    return ex
