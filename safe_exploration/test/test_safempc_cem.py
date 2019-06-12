import numpy as np
import torch
from constrained_cem_mpc import ActionConstraint

from .. import safempc_cem
from ..environments import CartPole, InvertedPendulum
from ..safempc_cem import PQFlattener, EllipsoidTerminalConstraint, CemSafeMPC


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
    constraints = safempc_cem.construct_constraints(env)
    assert len([c for c in constraints if isinstance(c, EllipsoidTerminalConstraint)]) == 1


def test__construct_constraints__contains_one_action_constraint():
    # TODO: check constraint is actually correct.
    env = CartPole()
    constraints = safempc_cem.construct_constraints(env)
    assert len([c for c in constraints if isinstance(c, ActionConstraint)]) == 1


class TestCemSafeMPC:
    @staticmethod
    def _get_opt_env():
        return {'lin_model': ([0.1, 0.2])}

    def test__get_action__mpc_has_solution__returns_first_action(self, mocker):
        ssm = mocker.Mock()
        mpc = mocker.Mock()
        mpc.get_actions.return_value = (torch.tensor([[0.1], [0.2]]), [])
        safe_mpc = CemSafeMPC(ssm, [], InvertedPendulum(), self._get_opt_env(), wx_feedback_cost=None, wu_feedback_cost=None,
                              mpc_time_horizon=2, plot_cem_optimisation=False, lqr=mocker.Mock(), mpc=mpc)
        safe_mpc.update_model(np.array([[0.1, 0.2, 0.3]]), np.array([[0.1, 0.1]]))

        action, success = safe_mpc.get_action(np.array([0., 0.]))

        assert np.allclose(action, np.array([0.1]))

    def test__get_action__previous_mpc_solution__returns_next_action_from_previous_solution(self, mocker):
        ssm = mocker.Mock()
        mpc = mocker.Mock()
        mpc.get_actions.side_effect = [(torch.tensor([[0.1], [0.2]]), []), (None, [])]
        safe_mpc = CemSafeMPC(ssm, [], InvertedPendulum(), self._get_opt_env(), wx_feedback_cost=None, wu_feedback_cost=None,
                              mpc_time_horizon=2, plot_cem_optimisation=False, lqr=mocker.Mock(), mpc=mpc)
        safe_mpc.update_model(np.array([[0.1, 0.2, 0.3]]), np.array([[0.1, 0.1]]))

        safe_mpc.get_action(np.array([0., 0.]))
        action, success = safe_mpc.get_action(np.array([0., 0.]))

        assert np.allclose(action, np.array([0.2]))

    def test__get_action__no_previous_mpc_solution__returns_safe_action(self, mocker):
        ssm = mocker.Mock()

        lqr = mocker.Mock()
        lqr.get_control_matrix.return_value = np.eye(2)

        mpc = mocker.Mock()
        mpc.get_actions.side_effect = [(None, [])]

        safe_mpc = CemSafeMPC(ssm, [], InvertedPendulum(), self._get_opt_env(), wx_feedback_cost=None, wu_feedback_cost=None,
                              mpc_time_horizon=2, plot_cem_optimisation=False, lqr=lqr, mpc=mpc)
        safe_mpc.update_model(np.array([[0.1, 0.2, 0.3]]), np.array([[0.1, 0.1]]))

        action, success = safe_mpc.get_action(np.array([1., 2.]))

        assert np.allclose(action, np.array([1., 2.]))

    def test__get_action__previous_solution_run_out__returns_safe_action(self, mocker):
        ssm = mocker.Mock()

        lqr = mocker.Mock()
        lqr.get_control_matrix.return_value = np.eye(2)

        mpc = mocker.Mock()
        mpc.get_actions.side_effect = [(torch.tensor([[0.1], [0.2]]), []), (None, []), (None, [])]

        safe_mpc = CemSafeMPC(ssm, [], InvertedPendulum(), self._get_opt_env(), wx_feedback_cost=None, wu_feedback_cost=None,
                              mpc_time_horizon=2, plot_cem_optimisation=False, lqr=lqr, mpc=mpc)
        safe_mpc.update_model(np.array([[0.1, 0.2, 0.3]]), np.array([[0.1, 0.1]]))

        safe_mpc.get_action(np.array([0., 0.]))
        safe_mpc.get_action(np.array([0., 0.]))
        action, success = safe_mpc.get_action(np.array([1., 2.]))

        assert np.allclose(action, np.array([1., 2.]))
