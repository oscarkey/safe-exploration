import torch

from ..ssm_cem import GpCemSSM, McDropoutSSM


class TestGpCemSSM:
    class FakeConfig:
        exact_gp_training_iterations = 200
        device = None

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


class TestMcDropoutSSM:
    class FakeConfig:
        mc_dropout_training_iterations = 1000
        mc_dropout_hidden_features = [2, 2]
        mc_dropout_num_samples = 12
        mc_dropout_predict_std = True
        device = None

    def test__predict_without_jacobians__returns_correct_shape(self):
        ssm = McDropoutSSM(TestMcDropoutSSM.FakeConfig(), state_dimen=2, action_dimen=1)
        states = torch.tensor([[1., 1.], [1., 1.], [1., 1.]])
        actions = torch.tensor([[1.], [1.], [1.]])

        mean, var = ssm.predict_without_jacobians(states, actions)

        assert mean.size() == (3, 2)
        assert var.size() == (3, 2)

    def test__predict_with_jacobians__returns_correct_shape(self):
        ssm = McDropoutSSM(TestMcDropoutSSM.FakeConfig(), state_dimen=2, action_dimen=1)
        states = torch.tensor([[1., 1.], [1., 1.], [1., 1.]])
        actions = torch.tensor([[1.], [1.], [1.]])

        mean, var, jac = ssm.predict_with_jacobians(states, actions)

        assert mean.size() == (3, 2)
        assert var.size() == (3, 2)
        assert jac.size() == (3, 2, 2 + 1)
