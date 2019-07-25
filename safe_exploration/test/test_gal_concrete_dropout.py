from ..ssm_cem.gal_concrete_dropout import GalConcreteDropoutSSM


class TestGalConcreteDropoutSSM:
    class FakeConfig:
        mc_dropout_training_iterations = 4
        mc_dropout_hidden_features = [2, 2]
        mc_dropout_num_samples = 12
        mc_dropout_predict_std = True
        mc_dropout_reinitialize = False
        mc_dropout_type = 'concrete'
        mc_dropout_concrete_initial_probability = 0.1
        mc_dropout_fixed_probability = 0.1
        mc_dropout_on_input = True
        mc_dropout_lengthscale = 1e-4
        device = None

    def test__collect_metrics__returns_dropout_p_per_layer(self):
        conf = TestGalConcreteDropoutSSM.FakeConfig()
        conf.mc_dropout_hidden_features = [3, 2]
        ssm = GalConcreteDropoutSSM(conf, state_dimen=3, action_dimen=2)

        metrics = ssm.collect_metrics()

        dropout_keys = [key for key in metrics.keys() if key.startswith('dropout_p')]
        assert len(dropout_keys) == 4
