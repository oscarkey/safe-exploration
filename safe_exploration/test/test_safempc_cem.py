from safempc_cem import CemSafeMPC


class TestCemSafeMPC:
    def test__init__does_not_crash(self):
        CemSafeMPC(state_dimen=2, action_dimen=2)
