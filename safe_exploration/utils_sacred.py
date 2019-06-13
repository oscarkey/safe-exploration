from collections import defaultdict
from typing import Dict, List


class SacredAggregatedMetrics:
    """Collects metrics over a series of experiments. Logs the complete metrics, and the mean to Sacred."""

    def __init__(self, _run):
        self._run = _run
        # Values are a [key][counter] = list of values
        self._metrics = defaultdict(lambda: defaultdict(lambda: []))

    def log_scalar(self, metric_name, value, counter):
        """Add metric_name=value at t=counter to the logs. Does not send logs to Sacred, call flush() for this."""
        self._metrics[metric_name][counter].append(value)

    def flush(self):
        """Uploads the metrics in the buffer, and their aggregation, to Sacred. Then clears the buffer."""
        self._upload_metric_means()
        self._run.info['all_metrics'] = self._default_dict_to_dict(self._metrics)
        self._metrics.clear()
        print('Uploaded metrics to Sacred')

    def _upload_metric_means(self) -> None:
        for metric_name, by_counter in self._compute_means(self._metrics).items():
            for t, value in by_counter.items():
                self._run.log_scalar(metric_name, value, t)

    @staticmethod
    def _compute_means(metrics: Dict[str, Dict[int, List[float]]]) -> Dict[str, Dict[int, float]]:
        mean_metrics = defaultdict(lambda: {})
        for metric_name, by_counter in metrics.items():
            for t, values in by_counter.items():
                mean_metrics[metric_name][t] = sum(values) / len(values)
        return mean_metrics

    @staticmethod
    def _default_dict_to_dict(d):
        if isinstance(d, defaultdict):
            return {k: SacredAggregatedMetrics._default_dict_to_dict(v) for k, v in d.items()}
        else:
            return d
