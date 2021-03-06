import numpy as np
import torch
from constrained_cem_mpc import ActionConstraint
from numpy import ndarray

from .. import safempc_cem
from ..environments.environments import CartPole, InvertedPendulum
from ..safempc_cem import PQFlattener, EllipsoidTerminalConstraint, CemSafeMPC, MpcResult


class FakeConfig:
    mpc_time_horizon = 2
    cem_num_rollouts = 20
    cem_num_elites = 3
    cem_num_iterations = 8
    plot_cem_optimisation = False
    plot_cem_terminal_states = False
    device = None
    use_state_constraint = False
    use_prior_model = True


class TestPQFlattener:
    def test__flatten_unflatten__q_not_none__input_equals_output(self):
        flattener = PQFlattener(state_dimen=3)
        p = torch.tensor([[1, 2, 3], [10, 20, 30]])
        q = torch.tensor([[[10, 11, 12], [20, 21, 22], [30, 31, 32]], [[20, 21, 22], [30, 31, 32], [50, 51, 52]]])
        p_out, q_out = flattener.unflatten(flattener.flatten(p, q))
        assert torch.allclose(p, p_out)
        assert torch.allclose(q, q_out)

    def test__flatten_unflatten__q_is_none__input_equals_output(self):
        flattener = PQFlattener(state_dimen=4)
        p = torch.tensor([[1, 2, 3, 4], [10, 20, 30, 40]])
        q = None
        p_out, q_out = flattener.unflatten(flattener.flatten(p, q))
        assert torch.allclose(p, p_out)
        assert q_out is None

    def test__get_flat_state_dimen__returns_size_of_p_q_flattened(self):
        flattener = PQFlattener(state_dimen=5)
        assert flattener.get_flat_state_dimen() == 5 + 5 * 5


def test__construct_constraints__contains_one_terminal_constraint():
    # TODO: check constraint is actually correct.
    env = CartPole()
    constraints = safempc_cem.construct_constraints(FakeConfig(), env)
    assert len([c for c in constraints if isinstance(c, EllipsoidTerminalConstraint)]) == 1


def test__construct_constraints__contains_one_action_constraint():
    # TODO: check constraint is actually correct.
    env = CartPole()
    constraints = safempc_cem.construct_constraints(FakeConfig(), env)
    assert len([c for c in constraints if isinstance(c, ActionConstraint)]) == 1


def test__construct_constraints__action_constraint_correct():
    env = CartPole(u_min=np.array([-4.0]), u_max=np.array([4.0]))

    constraints = safempc_cem.construct_constraints(FakeConfig(), env)
    constraint = [c for c in constraints if isinstance(c, ActionConstraint)][0]

    N = 3
    trajectory = torch.zeros((N, env.n_s), dtype=torch.double)
    actions = torch.tensor([[0.2], [-5.], [6.]], dtype=torch.double)
    penalty = constraint(trajectory, actions)

    # Two constraints violated, each of cost 3.
    assert penalty == 2 * 3


class TestCemSafeMPC:
    @staticmethod
    def _safe_policy(x: ndarray) -> ndarray:
        return np.dot(x, np.eye(2))

    @staticmethod
    def _get_opt_env():
        return {'lin_model': ([0.1, 0.2])}

    def test__get_action__mpc_has_solution__returns_first_action(self, mocker):
        ssm = mocker.Mock()
        mpc = mocker.Mock()
        mpc.get_actions.return_value = (torch.tensor([[0.1], [0.2]]), [])
        safe_mpc = CemSafeMPC(ssm, [], InvertedPendulum(), FakeConfig(), self._get_opt_env(), wx_feedback_cost=None,
                              wu_feedback_cost=None, lqr=mocker.Mock(), mpc=mpc, beta_safety=1.0,
                              safe_policy=self._safe_policy)
        safe_mpc.update_model(np.array([[0.1, 0.2, 0.3]]), np.array([[0.1, 0.1]]))

        action, result = safe_mpc.get_action(np.array([0., 0.]))

        assert np.allclose(action, np.array([0.1]))
        assert result == MpcResult.FOUND_SOLUTION

    def test__get_action__previous_mpc_solution__returns_next_action_from_previous_solution(self, mocker):
        ssm = mocker.Mock()
        mpc = mocker.Mock()
        mpc.get_actions.side_effect = [(torch.tensor([[0.1], [0.2]]), []), (None, [])]
        safe_mpc = CemSafeMPC(ssm, [], InvertedPendulum(), FakeConfig(), self._get_opt_env(), wx_feedback_cost=None,
                              wu_feedback_cost=None, lqr=mocker.Mock(), mpc=mpc, beta_safety=1.0,
                              safe_policy=self._safe_policy)
        safe_mpc.update_model(np.array([[0.1, 0.2, 0.3]]), np.array([[0.1, 0.1]]))

        safe_mpc.get_action(np.array([0., 0.]))
        action, result = safe_mpc.get_action(np.array([0., 0.]))

        assert np.allclose(action, np.array([0.2]))
        assert result == MpcResult.PREVIOUS_SOLUTION

    def test__get_action__no_previous_mpc_solution__returns_safe_action(self, mocker):
        ssm = mocker.Mock()

        lqr = mocker.Mock()
        lqr.get_control_matrix.return_value = np.eye(2)

        mpc = mocker.Mock()
        mpc.get_actions.side_effect = [(None, [])]

        safe_mpc = CemSafeMPC(ssm, [], InvertedPendulum(), FakeConfig(), self._get_opt_env(), wx_feedback_cost=None,
                              wu_feedback_cost=None, lqr=lqr, mpc=mpc, beta_safety=1.0, safe_policy=self._safe_policy)
        safe_mpc.update_model(np.array([[0.1, 0.2, 0.3]]), np.array([[0.1, 0.1]]))

        action, result = safe_mpc.get_action(np.array([1., 2.]))

        assert np.allclose(action, np.array([1., 2.]))
        assert result == MpcResult.SAFE_CONTROLLER

    def test__get_action__previous_solution_run_out__returns_safe_action(self, mocker):
        ssm = mocker.Mock()

        lqr = mocker.Mock()
        lqr.get_control_matrix.return_value = np.eye(2)

        mpc = mocker.Mock()
        mpc.get_actions.side_effect = [(torch.tensor([[0.1], [0.2]]), []), (None, []), (None, [])]

        safe_mpc = CemSafeMPC(ssm, [], InvertedPendulum(), FakeConfig(), self._get_opt_env(), wx_feedback_cost=None,
                              wu_feedback_cost=None, lqr=lqr, mpc=mpc, beta_safety=1.0, safe_policy=self._safe_policy)
        safe_mpc.update_model(np.array([[0.1, 0.2, 0.3]]), np.array([[0.1, 0.1]]))

        safe_mpc.get_action(np.array([0., 0.]))
        safe_mpc.get_action(np.array([0., 0.]))
        action, result = safe_mpc.get_action(np.array([1., 2.]))

        assert np.allclose(action, np.array([1., 2.]))
        assert result == MpcResult.SAFE_CONTROLLER
