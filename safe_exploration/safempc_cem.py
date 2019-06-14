from typing import Optional, Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from constrained_cem_mpc import ConstrainedCemMpc, ActionConstraint, box2torchpoly, Constraint, Rollouts, DynamicsFunc
from constrained_cem_mpc.utils import assert_shape
from numpy import ndarray
from polytope import Polytope
from torch import Tensor

from . import gp_reachability_pytorch
from .environments import Environment
from .gp_reachability_pytorch import onestep_reachability
from .safempc import SafeMPC
from .safempc_simple import LqrFeedbackController
from .ssm_cem import CemSSM
from .utils import get_pytorch_device
from .visualization import utils_visualization


class PQFlattener:
    """Converts ellipsoid p and q matrices to a flat state vector, and vice versa.

    As each state is an ellipsoid it has both a centre (p) and a shape (q). However, constrained-cem-mpc currently
    requires the state to be a single vector, so this class handles the conversion between the two.
    """

    def __init__(self, state_dimen: int):
        self._state_dimen = state_dimen

    def flatten(self, p: Tensor, q: Optional[Tensor]) -> Tensor:
        """Converts p and q into a single state tensor.

        :param p: [N x state dimen], batch of p vectors
        :param q: [N x state dimen x state dimen], batch of q matricies, or None if all the states are points
        :returns: [N x (state dimen + state_dimen * state_dimen], batch of flattened states
        """
        N = p.size(0)

        if q is None:
            q = torch.zeros((N, self._state_dimen, self._state_dimen), dtype=p.dtype, device=p.device)

        assert_shape(p, (N, self._state_dimen,))
        assert_shape(q, (N, self._state_dimen, self._state_dimen))
        return torch.cat((p.reshape(N, -1), q.reshape(N, -1)), dim=1)

    def unflatten(self, flat: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Converts a flat state into p and q.

        :param flat: [N x (state dimen + state_dimen * state_dimen], batch of flattened states
        :returns:
            p: [N x state dimen], batch of p vectors,
            q: [N x state dimen x state dimen], batch of q matricies, or None if all the states are points
        """
        N = flat.size(0)
        assert_shape(flat, (N, self.get_flat_state_dimen(),))
        p = flat[:, 0:self._state_dimen]
        q = flat[:, self._state_dimen:].view(N, self._state_dimen, self._state_dimen)

        # If q is all zeros, we treat this as None.
        if len(q.nonzero()) == 0:
            q = None

        return p, q

    def get_flat_state_dimen(self):
        return self._state_dimen + (self._state_dimen * self._state_dimen)


def _plot_constraints_in_2d(h_mat_safe, h_safe, h_mat_obs, h_obs) -> None:
    ax = plt.axes()
    # Hacky way to test if cartpole or pendulum.
    num_cartpole_states = 4
    if h_mat_safe.shape[1] == num_cartpole_states:
        Polytope(h_mat_obs[:, [0, 2]], h_obs).plot(ax=ax, color='blue')
        Polytope(h_mat_safe[:, [0, 2]], h_safe).plot(ax=ax, color='red')
    else:
        Polytope(h_mat_safe, h_safe).plot(ax=ax, color='grey')


def _plot_ellipsoids_in_2d(p: Tensor, q: Tensor, color: Union[str, Tuple[float, float, float]] = 'orange') -> None:
    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()
    # Hacky way to test if cartpole or pendulum.
    num_cartpole_states = 4
    if p.shape[0] == num_cartpole_states:
        utils_visualization.plot_ellipsoid_2D(p[[0, 2], :], np.delete(np.delete(q, [1, 2], 0), [1, 2], 1), ax=plt.gca(),
                                              color=color, n_points=50)
    else:
        utils_visualization.plot_ellipsoid_2D(p, q, ax=plt.gca(), color=color, n_points=50)


class EllipsoidTerminalConstraint(Constraint):
    """Represents the terminal constraint of the MPC problem, for ellipsoid states and a polytopic constraint."""

    def __init__(self, state_dimen: int, safe_polytope_a: np.ndarray, safe_polytope_b: np.ndarray):
        self._pq_flattener = PQFlattener(state_dimen)
        self._polytope_a = torch.tensor(safe_polytope_a, device=get_pytorch_device())
        self._polytope_b = torch.tensor(safe_polytope_b, device=get_pytorch_device())

    def __call__(self, trajectory, actions) -> float:
        # Add batch dimension.
        trajectory = trajectory.unsqueeze(0)

        p, q = self._pq_flattener.unflatten(trajectory[:, -1])

        # Remove batch dimension.
        p = p.squeeze(0)
        q = q.squeeze(0) if q is not None else q

        p = p.unsqueeze(1)

        if gp_reachability_pytorch.is_ellipsoid_inside_polytope(p, q, self._polytope_a, self._polytope_b):
            return 0
        else:
            return 10


def construct_constraints(env: Environment):
    """Creates the polytopic constraints for the MPC problem from the values in the config file."""
    h_mat_safe, h_safe, h_mat_obs, h_obs = env.get_safety_constraints(normalize=True)
    action_constraint = ActionConstraint(
        box2torchpoly([[env.u_min.item(), env.u_max.item()], ]).to(get_pytorch_device()))
    terminal_constraint = EllipsoidTerminalConstraint(env.n_s, h_mat_safe, h_safe)
    return [action_constraint, terminal_constraint]


class DynamicsFuncWrapper(DynamicsFunc):
    """Wraps a given function as a DynamicsFunc."""

    def __init__(self, func):
        self._func = func

    def __call__(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        return self._func(states, actions)


class CemSafeMPC(SafeMPC):
    """Safe MPC implementation which uses the constrained CEM to optimise the trajectories."""

    def __init__(self, ssm: CemSSM, constraints: [Constraint], env: Environment, conf, opt_env, wx_feedback_cost,
                 wu_feedback_cost, lqr: Optional[LqrFeedbackController] = None,
                 mpc: Optional[ConstrainedCemMpc] = None) -> None:
        super().__init__()

        self._state_dimen = env.n_s
        self._action_dimen = env.n_u
        self._l_mu = torch.tensor(env.l_mu, device=get_pytorch_device())
        self._l_sigma = torch.tensor(env.l_sigm, device=get_pytorch_device())
        self._get_random_action = env.random_action
        self._pq_flattener = PQFlattener(env.n_s)
        self._ssm = ssm
        self._plot = conf.plot_cem_optimisation
        self._mpc_time_horizon = conf.mpc_time_horizon

        linearized_model_a, linearized_model_b = opt_env['lin_model']
        self.lin_model = opt_env['lin_model']
        self._linearized_model_a = torch.tensor(linearized_model_a, device=get_pytorch_device())
        self._linearized_model_b = torch.tensor(linearized_model_b, device=get_pytorch_device())

        if lqr is None:
            lqr = LqrFeedbackController(wx_feedback_cost, wu_feedback_cost, env.n_s, env.n_u, linearized_model_a,
                                        linearized_model_b)
        self._lqr = lqr

        if mpc is None:
            mpc = ConstrainedCemMpc(DynamicsFuncWrapper(self._dynamics_func), constraints=constraints,
                                    state_dimen=self._pq_flattener.get_flat_state_dimen(), action_dimen=env.n_u,
                                    time_horizon=self._mpc_time_horizon, num_rollouts=conf.cem_num_rollouts,
                                    num_elites=conf.cem_num_elites, num_iterations=conf.cem_num_iterations)
        self._mpc = mpc

        self._has_training_data = False

        self._last_mpc_actions = np.empty((0, self.action_dimen))
        self._mpc_actions_executed = 0

    @property
    def state_dimen(self) -> int:
        return self._state_dimen

    @property
    def action_dimen(self) -> int:
        return self._action_dimen

    @property
    def safety_trajectory_length(self) -> int:
        return self._mpc_time_horizon

    @property
    def performance_trajectory_length(self) -> int:
        # We haven't yet implemented performance trajectories in CEM.
        return 0

    @property
    def x_train(self) -> ndarray:
        x_train = self._ssm.x_train
        if x_train is None:
            return np.empty((0, self._state_dimen + self._action_dimen))
        else:
            return x_train.detach().cpu().numpy()

    def init_solver(self, cost_func=None):
        # TODO: attach the cost function to the mpc.
        pass

    def get_action(self, state: ndarray) -> Tuple[ndarray, bool]:
        assert_shape(state, (self._state_dimen,))

        # If we don't have training data we skip solving the mpc as it won't be any use.
        # This makes the first episode much faster (during which we gather training data)
        if self._has_training_data:
            state_batch = torch.tensor(state, device=get_pytorch_device()).unsqueeze(0)
            mpc_actions, rollouts = self._mpc.get_actions(self._pq_flattener.flatten(state_batch, None))
            mpc_actions = mpc_actions.detach().cpu().numpy() if mpc_actions is not None else mpc_actions

            if self._plot:
                self._plot_rollouts(rollouts)
        else:
            print('No training data')
            mpc_actions = None

        # TODO: Need to maintain a queue of previously safe actions to fully implement SafeMPC.

        if mpc_actions is not None:
            print('Found solution')
            success = True
            self._last_mpc_actions = mpc_actions
            self._mpc_actions_executed = 0
        else:
            print('No solution found')
            success = False

        if self._mpc_actions_executed < self._last_mpc_actions.shape[0]:
            print('Using existing solution')
            action = self._last_mpc_actions[self._mpc_actions_executed]
            self._mpc_actions_executed += 1
        else:
            print('Using safe controller')
            action = self._get_safe_controller_action(state)

        return action, success

    def get_action_verbose(self, state: ndarray):
        raise NotImplementedError

    def _plot_rollouts(self, rollouts: List[Rollouts]):
        """Plots the constraint, and the terminal states of the rollouts through the optimisation process."""
        tc = self._mpc._rollout_function._constraints[1]
        _plot_constraints_in_2d(tc._polytope_a.detach().cpu().numpy(), tc._polytope_b.detach().cpu().numpy(), None,
                                None)
        for i in range(len(rollouts)):
            ps, qs = self._pq_flattener.unflatten(rollouts[i].trajectories[:, -1, :])
            for j in range(ps.size(0)):
                red = hex(round(255.0 / self._mpc._num_iterations * i))[2:]
                if red == '0':
                    red = '00'
                color = f'#{red}0000'
                _plot_ellipsoids_in_2d(ps[j].unsqueeze(1), qs[j], color=color)

        plt.show()

    def _dynamics_func(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        ps, qs = self._pq_flattener.unflatten(states)

        p_next, q_next, sigma = onestep_reachability(ps, self._ssm, actions, self._l_mu, self._l_sigma, qs,
                                                     k_fb=self._lqr.get_control_matrix_pytorch(),
                                                     a=self._linearized_model_a, b=self._linearized_model_b, verbose=0)

        # Try to maximise the variance in the predictions so we explore as must as possible.
        objective_cost = - torch.sum((sigma), dim=1)

        return self._pq_flattener.flatten(p_next, q_next), objective_cost

    def _get_safe_controller_action(self, state):
        # TODO: Use the safe policy from the config (though I think this is actually always just lqr)
        return np.dot(self._lqr.get_control_matrix(), state)

    def update_model(self, x: ndarray, y: ndarray, opt_hyp=False, replace_old=True, reinitialize_solver=True) -> None:
        x = torch.tensor(x, device=get_pytorch_device())
        y = torch.tensor(y, device=get_pytorch_device())
        self._ssm.update_model(x, y, opt_hyp, replace_old)
        self._has_training_data = True

    def information_gain(self) -> Union[ndarray, List[None]]:
        print('Not implemented')
        return [None] * self.state_dimen

    def ssm_predict(self, z: ndarray) -> Tuple[ndarray, ndarray]:
        mean, sigma = self._ssm.predict_raw(torch.tensor(z, device=get_pytorch_device()))
        return mean.detach().cpu().numpy(), sigma.detach().cpu().numpy()

    def eval_prior(self, states: ndarray, actions: ndarray):
        a = self._linearized_model_a.cpu().numpy()
        b = self._linearized_model_b.cpu().numpy()
        return np.dot(states, a.T) + np.dot(actions, b.T)
