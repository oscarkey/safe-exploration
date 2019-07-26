import os
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
from matplotlib.figure import Figure
from numpy import ndarray
from sacred.run import Run


class SacredAggregatedMetrics:
    """Collects metrics over a series of experiments. Logs the complete metrics, and the mean to Sacred."""

    def __init__(self, _run: Run):
        self._run = _run
        # Values are a [key][counter] = list of values
        self._aggregated_metrics = defaultdict(lambda: defaultdict(lambda: []))
        self._non_aggregated_metrics = defaultdict(lambda: defaultdict(lambda: []))

    def log_scalar(self, metric_name, value, counter):
        """Add metric_name=value at t=counter to the logs. Does not send logs to Sacred, call flush() for this."""
        if metric_name in self._non_aggregated_metrics:
            raise ValueError(f'{metric_name} already logged as a non-scalar metric')

        self._aggregated_metrics[metric_name][counter].append(value)

    def log_scalars(self, metrics: Dict[str, float], counter):
        """Adds a set of metric_name=value pairs at t=counter to the logs.

        Does not send the logs to Sacred, call flush() for this.
        """
        for k, v in metrics.items():
            self.log_scalar(k, v, counter)

    def log_non_scalar(self, metric_name, value, counter):
        """Logs metric_name=value at t=counter. As 'value' is non-scalar, it will not be aggregated across scenarios.

        Does not send the logs to Sacred, call flush() for this.
        """
        if metric_name in self._aggregated_metrics:
            raise ValueError(f'{metric_name} already logged as a scalar metric')

        self._non_aggregated_metrics[metric_name][counter].append(value)

    def log_non_scalars(self, metrics: Dict[str, Any], counter):
        """Logs a set of metric_name=value pairs at t=counter.

        As 'value' is non-scalar, it will not be aggregated across scenarios.
        Does not send the logs to Sacred, call flush() for this.
        """
        for k, v in metrics.items():
            self.log_non_scalar(k, v, counter)

    def flush(self):
        """Uploads the metrics in the buffer, and their aggregation, to Sacred. Then clears the buffer."""
        self._upload_metric_means()
        self._run.info['all_metrics'] = {**self._default_dict_to_dict(self._aggregated_metrics),
                                         **self._default_dict_to_dict(self._non_aggregated_metrics)}
        self._aggregated_metrics.clear()
        self._non_aggregated_metrics.clear()
        print('Uploaded metrics to Sacred')

    def _upload_metric_means(self) -> None:
        for metric_name, by_counter in self._compute_means(self._aggregated_metrics).items():
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

    def save_figure(self, figure: Figure, name: str):
        """Saves the given figure to a file, and adds it as an artifact to sacred."""
        dir_name = self._get_artifacts_dir()
        file_name = f'{self._run._id}_{name}.png'
        file_path = os.path.join(dir_name, file_name)
        figure.savefig(file_path)
        self._run.add_artifact(file_path)

    def save_array(self, array: ndarray, name: str):
        """Saves the given array as a ffile, and adds it as an artifact to sacred."""
        dir_name = self._get_artifacts_dir()
        file_name = f'{self._run._id}_{name}'
        file_path = os.path.join(dir_name, file_name)
        np.save(file_path, array)
        self._run.add_artifact(file_path)

    @staticmethod
    def _get_artifacts_dir() -> str:
        dir_name = 'safe_exploration_results/artifacts'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        return dir_name
