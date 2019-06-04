import numpy as np
import torch
from constrained_cem_mpc import ActionConstraint

from .. import safempc_cem
from ..environments import CartPole
from ..safempc_cem import PQFlattener, EllipsoidTerminalConstraint


class TestPQFlattener:
    def test__flatten_unflatten__q_not_none__input_equals_output(self):
        flattener = PQFlattener(state_dimen=3)
        p = torch.tensor([1, 2, 3])
        q = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
        p_out, q_out = flattener.unflatten(flattener.flatten(p, q))
        assert torch.allclose(p, p_out)
        assert torch.allclose(q, q_out)

    def test__flatten_unflatten__q_is_none__input_equals_output(self):
        flattener = PQFlattener(state_dimen=4)
        p = torch.tensor([1, 2, 3, 4])
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
