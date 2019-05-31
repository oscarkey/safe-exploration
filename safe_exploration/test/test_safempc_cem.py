import numpy as np

from ..safempc_cem import CemSafeMPC, PQFlattener


class TestPQFlattener:
    def test__flatten_unflatten__q_not_none__input_equals_output(self):
        flattener = PQFlattener(state_dimen=3)
        p = np.array([1, 2, 3])
        q = np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
        p_out, q_out = flattener.unflatten(flattener.flatten(p, q))
        assert np.array_equal(p, p_out)
        assert np.array_equal(q, q_out)

    def test__flatten_unflatten__q_is_none__input_equals_output(self):
        flattener = PQFlattener(state_dimen=4)
        p = np.array([1, 2, 3, 4])
        q = None
        p_out, q_out = flattener.unflatten(flattener.flatten(p, q))
        assert np.array_equal(p, p_out)
        assert q_out is None

    def test__get_flat_state_dimen__returns_size_of_p_q_flattened(self):
        flattener = PQFlattener(state_dimen=5)
        assert flattener.get_flat_state_dimen() == 5 + 5 * 5


class TestCemSafeMPC:
    def test__init__does_not_crash(self):
        CemSafeMPC(state_dimen=2, action_dimen=2)
