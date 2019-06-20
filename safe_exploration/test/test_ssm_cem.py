import random

import numpy as np
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
        mc_dropout_training_iterations = 4
        mc_dropout_hidden_features = [2, 2]
        mc_dropout_num_samples = 12
        mc_dropout_predict_std = True
        mc_dropout_reinitialize = False
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

    def test__update_model__reinitialize_on_train_false__does_not_reinitialize_weights(self):
        config = TestMcDropoutSSM.FakeConfig()
        config.mc_dropout_reinitialize = False
        config.mc_dropout_training_iterations = 0

        states = torch.tensor([[1., 1.], [1., 1.], [1., 1.]])
        actions = torch.tensor([[1.], [1.], [1.]])

        # If reinitialize=False, then we should get the same result whether we call train or not. We set training
        # iterations to 0 so train() does not optimise the weights.

        self._seed_rng()
        ssm = McDropoutSSM(config, state_dimen=2, action_dimen=1)
        mean1, var1 = ssm.predict_without_jacobians(states, actions)

        self._seed_rng()
        ssm = McDropoutSSM(config, state_dimen=2, action_dimen=1)
        ssm.update_model(torch.empty((0, 3)), torch.empty((0, 2)), opt_hyp=True)
        mean2, var2 = ssm.predict_without_jacobians(states, actions)

        assert torch.allclose(mean1, mean2)

    def test__update_model__reinitialize_on_train_true__reinitializes_weights(self):
        config = TestMcDropoutSSM.FakeConfig()
        config.mc_dropout_reinitialize = True
        config.mc_dropout_training_iterations = 0

        states = torch.tensor([[1., 1.], [1., 1.], [1., 1.]])
        actions = torch.tensor([[1.], [1.], [1.]])

        # If reinitialize=True, then we should get a different result if we have called train() as this will have
        # reinitialized the weights.

        self._seed_rng()
        ssm = McDropoutSSM(config, state_dimen=2, action_dimen=1)
        mean1, var1 = ssm.predict_without_jacobians(states, actions)

        self._seed_rng()
        ssm = McDropoutSSM(config, state_dimen=2, action_dimen=1)
        ssm.update_model(torch.empty((0, 3)), torch.empty((0, 2)), opt_hyp=True)
        mean2, var2 = ssm.predict_without_jacobians(states, actions)

        assert not torch.allclose(mean1, mean2)

    @staticmethod
    def _seed_rng():
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
