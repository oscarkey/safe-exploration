import random

import numpy as np
import torch

from ..ssm_cem.dropout_ssm_cem import McDropoutSSM
from ..ssm_cem.ssm_cem import JunkDimensionsSSM


class TestMcDropoutSSM:
    class FakeConfig:
        mc_dropout_training_iterations = 4
        mc_dropout_hidden_features = [2, 2]
        mc_dropout_num_samples = 12
        mc_dropout_predict_std = True
        mc_dropout_reinitialize = False
        mc_dropout_type = 'concrete'
        mc_dropout_concrete_initial_probability = 0.1
        mc_dropout_fixed_probability = 0.1
        mc_dropout_on_input = False
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

    def test__collect_metrics__no_input_dropout__returns_dropout_p_per_layer(self):
        config = TestMcDropoutSSM.FakeConfig()
        config.mc_dropout_hidden_features = [2, 2, 10, 2]
        config.mc_dropout_on_input = False
        ssm = McDropoutSSM(config, state_dimen=2, action_dimen=1)

        ps = ssm.collect_metrics()

        assert ps.keys() == {'dropout_p_layer_1', 'dropout_p_layer_4', 'dropout_p_layer_7', 'dropout_p_layer_10'}

    def test__collect_metrics__input_dropout__returns_dropout_p_per_layer(self):
        config = TestMcDropoutSSM.FakeConfig()
        config.mc_dropout_hidden_features = [2, 2, 10, 2]
        config.mc_dropout_on_input = True
        ssm = McDropoutSSM(config, state_dimen=2, action_dimen=1)

        ps = ssm.collect_metrics()

        assert ps.keys() == {'dropout_p_layer_0', 'dropout_p_layer_2', 'dropout_p_layer_5', 'dropout_p_layer_8',
                             'dropout_p_layer_11'}

    @staticmethod
    def _seed_rng():
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)


class TestJunkDimensionsSSM:
    def test__predict_with_jacobians__expands_dimensions(self, mocker):
        inner_ssm = mocker.Mock()
        inner_ssm.predict_with_jacobians.return_value = (
            torch.empty((3, 7)), torch.empty((3, 7)), torch.empty((3, 7, 28)))
        constructor = mocker.Mock()
        constructor.return_value = inner_ssm

        ssm = JunkDimensionsSSM(constructor, state_dimen=2, action_dimen=1, junk_states=5, junk_actions=20)

        means, vars, jacs = ssm.predict_with_jacobians(torch.empty((3, 2)), torch.empty((3, 1)))

        assert means.size() == (3, 2)
        assert vars.size() == (3, 2)
        assert jacs.size() == (3, 2, 3)
        (call_states, call_actions), _ = inner_ssm.predict_with_jacobians.call_args
        assert call_states.size() == (3, 7)
        assert call_actions.size() == (3, 21)

    def test__predict_without_jacobians__expands_dimensions(self, mocker):
        inner_ssm = mocker.Mock()
        inner_ssm.predict_without_jacobians.return_value = (torch.empty((3, 7)), torch.empty((3, 7)))
        constructor = mocker.Mock()
        constructor.return_value = inner_ssm

        ssm = JunkDimensionsSSM(constructor, state_dimen=2, action_dimen=1, junk_states=5, junk_actions=20)

        means, vars = ssm.predict_without_jacobians(torch.empty((3, 2)), torch.empty((3, 1)))

        assert means.size() == (3, 2)
        assert vars.size() == (3, 2)
        (call_states, call_actions), _ = inner_ssm.predict_without_jacobians.call_args
        assert call_states.size() == (3, 7)
        assert call_actions.size() == (3, 21)

    def test__predict_raw__expands_dimensions(self, mocker):
        inner_ssm = mocker.Mock()
        inner_ssm.predict_raw.return_value = (torch.empty((3, 7)), torch.empty((3, 7)))
        constructor = mocker.Mock()
        constructor.return_value = inner_ssm

        ssm = JunkDimensionsSSM(constructor, state_dimen=2, action_dimen=1, junk_states=5, junk_actions=20)

        means, vars = ssm.predict_raw(torch.empty((3, 3)))

        assert means.size() == (3, 2)
        assert vars.size() == (3, 2)
        (call_z,), _ = inner_ssm.predict_raw.call_args
        assert call_z.size() == (3, 28)
