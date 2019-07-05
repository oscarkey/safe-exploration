from typing import Dict

from utils_sacred import SacredAggregatedMetrics


class TestSacredAggregatedMetrics:
    @staticmethod
    def _create_run(mocker) -> Dict:
        _run = mocker.Mock()
        _run.info = dict()
        return _run

    def test__log_scalar__does_not_crash(self, mocker):
        _run = self._create_run(mocker)
        metrics = SacredAggregatedMetrics(_run)

        metrics.log_scalar('metric1', 3.0, 1)

    def test__log_scalars__adds_metrics_to_queue(self, mocker):
        _run = self._create_run(mocker)
        metrics = SacredAggregatedMetrics(_run)

        collected_metrics_s1_e1 = {'metric1': 0.0, 'metric2': 100.0}

        collected_metrics_s1_e2 = {'metric1': 10.0, }

        collected_metrics_s2_e1 = {'metric1': 10.0, }

        collected_metrics_s2_e2 = {'metric1': 20.0, }

        metrics.log_scalars(collected_metrics_s1_e1, 1)
        metrics.log_scalars(collected_metrics_s1_e2, 1)
        metrics.log_scalars(collected_metrics_s2_e1, 2)
        metrics.log_scalars(collected_metrics_s2_e2, 2)

        metrics.flush()

        expected_metrics = {  #
            'metric1': {  #
                1: [0.0, 10.0],  #
                2: [10.0, 20.0]  #
            },  #
            'metric2': {  #
                1: [100.0]  #
            }  #
        }

        assert _run.info['all_metrics'] == expected_metrics

    def test__flush__uploads_means_to_sacred(self, mocker):
        _run = self._create_run(mocker)
        _run.info = {}
        metrics = SacredAggregatedMetrics(_run)

        metrics.log_scalar('metric1', 0.0, 1)
        metrics.log_scalar('metric1', 10.0, 1)
        metrics.log_scalar('metric1', 10.0, 2)
        metrics.log_scalar('metric1', 20.0, 2)
        metrics.log_scalar('metric2', 100.0, 1)

        metrics.flush()

        calls = _run.log_scalar.call_args_list
        assert len(calls) == 3
        assert calls[0][0] == ('metric1', 5.0, 1)
        assert calls[1][0] == ('metric1', 15.0, 2)
        assert calls[2][0] == ('metric2', 100.0, 1)

    def test__flush__adds_results_to_info_dict(self, mocker):
        _run = self._create_run(mocker)
        metrics = SacredAggregatedMetrics(_run)

        metrics.log_scalar('metric1', 0.0, 1)
        metrics.log_scalar('metric1', 10.0, 1)
        metrics.log_scalar('metric1', 10.0, 2)
        metrics.log_scalar('metric1', 20.0, 2)
        metrics.log_scalar('metric2', 100.0, 1)

        metrics.flush()

        expected_metrics = {  #
            'metric1': {  #
                1: [0.0, 10.0],  #
                2: [10.0, 20.0]  #
            },  #
            'metric2': {  #
                1: [100.0]  #
            }  #
        }

        assert _run.info['all_metrics'] == expected_metrics

    def test__flush__clears_buffer(self, mocker):
        _run = self._create_run(mocker)
        metrics = SacredAggregatedMetrics(_run)

        metrics.log_scalar('metric1', 0.0, 1)
        metrics.flush()
        metrics.log_scalar('metric2', 0.0, 1)
        metrics.flush()

        # TODO: This probably isn't actually what we want, we want to append to the info dict, but it will do for now.
        assert 'metric1' not in _run.info['all_metrics']
