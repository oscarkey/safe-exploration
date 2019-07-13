import torch

from ..ssm_cem.gp_ssm_cem import GpCemSSM


class TestGpCemSSM:
    class FakeConfig:
        exact_gp_training_iterations = 200
        device = None
        exact_gp_kernel = 'rbf'

    def test__x_train__no_training_data__returns_None(self):
        ssm = GpCemSSM(TestGpCemSSM.FakeConfig(), state_dimen=2, action_dimen=1)
        assert ssm.x_train is None

    def test__x_train__has_training_data__returns_data(self):
        ssm = GpCemSSM(TestGpCemSSM.FakeConfig(), state_dimen=2, action_dimen=1)
        xs = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        ys = torch.tensor([[20, 21], [30, 31], [40, 41], [50, 51]])
        ssm.update_model(xs, ys, opt_hyp=False, replace_old=True)

        x_train = ssm.x_train

        assert x_train.size() == (4, 3)
        assert torch.allclose(xs, x_train)

    def test__y_train__no_training_data__returns_None(self):
        ssm = GpCemSSM(TestGpCemSSM.FakeConfig(), state_dimen=2, action_dimen=1)
        assert ssm.y_train is None

    def test__y_train__has_training_data__returns_data(self):
        ssm = GpCemSSM(TestGpCemSSM.FakeConfig(), state_dimen=2, action_dimen=1)
        xs = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        ys = torch.tensor([[20, 21], [30, 31], [40, 41], [50, 51]])
        ssm.update_model(xs, ys, opt_hyp=False, replace_old=True)

        y_train = ssm.y_train

        assert y_train.size() == (4, 2)
        assert torch.allclose(ys, y_train)
