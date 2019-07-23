import math
from typing import Tuple, List, Dict, Any

import bnn
import numpy as np
import torch
from bnn import BDropout, CDropout
from torch import Tensor
from torch.nn import Module, functional as F

from .ssm_cem import CemSSM
from ..ssm_pytorch import utilities
from ..utils import get_device, assert_shape


class McDropoutSSM(CemSSM):
    """A BNN state space model, approximated using concrete mc dropout.

    Uses the "bnn" package from https://github.com/anassinator/bnn
    """

    def __init__(self, conf, state_dimen: int, action_dimen: int):
        super().__init__(state_dimen, action_dimen)

        self._training_iterations = conf.mc_dropout_training_iterations
        self._num_mc_samples = conf.mc_dropout_num_samples
        self._predict_std = conf.mc_dropout_predict_std
        self._reinitialize_on_train = conf.mc_dropout_reinitialize
        self._state_dimen = state_dimen

        self._model_constructor = self._get_model_constructor(conf, state_dimen, action_dimen)
        self._model = self._model_constructor()
        self._model.eval()

        self._loss_function = self._get_loss_function(conf)

    def _get_loss_function(self, conf):
        if conf.mc_dropout_type == 'fixed':
            return self._fixed_dropout_loss
        elif conf.mc_dropout_type == 'concrete':
            return self._concrete_dropout_loss
        else:
            raise ValueError(f'Unknown dropout type {conf.mc_dropout_type}')

    def _get_model_constructor(self, conf, state_dimen: int, action_dimen: int):
        in_features = state_dimen + action_dimen
        # Double the regression outputs. We need one for the mean and one for the predicted std (if enabled)
        out_features = state_dimen * 2 if self._predict_std else state_dimen

        def constructor() -> Module:
            input_dropout, dropout_layers = self._get_dropout_layers(conf)
            model = bnn.bayesian_model(in_features, out_features, hidden_features=conf.mc_dropout_hidden_features,
                                       dropout_layers=dropout_layers, input_dropout=input_dropout)
            model = model.to(get_device(conf))
            return model

        return constructor

    @staticmethod
    def _get_dropout_layers(conf) -> Tuple[Module, List[Module]]:
        hidden_features = conf.mc_dropout_hidden_features

        if conf.mc_dropout_type == 'fixed':
            p = conf.mc_dropout_fixed_probability
            input_dropout = BDropout(rate=p) if conf.mc_dropout_on_input else None
            dropout_layers = [BDropout(rate=p) for _ in hidden_features]

        elif conf.mc_dropout_type == 'concrete':
            p = conf.mc_dropout_concrete_initial_probability
            input_dropout = CDropout(rate=p) if conf.mc_dropout_on_input else None
            dropout_layers = [CDropout(rate=p) for _ in hidden_features]

        else:
            raise ValueError(f'Unknown dropout type {conf.mc_dropout_type}')

        return input_dropout, dropout_layers

    def predict_with_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z = self._join_states_actions(states, actions)
        pred_mean, pred_var = self.predict_raw(z)

        def mean_func(x: Tensor):
            return self.predict_raw(x)[0]

        pred_mean_jac = utilities.compute_jacobian_fast(mean_func, z, num_outputs=self.num_states)

        return pred_mean, pred_var, pred_mean_jac

    def predict_without_jacobians(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        z = self._join_states_actions(states, actions)
        return self.predict_raw(z)

    def predict_raw(self, z: Tensor):
        N = z.size(0)
        assert_shape(z, (N, self.num_states + self.num_actions))

        # To get the variance, we sample _num_particles times from the network.
        z_particles = z.repeat((self._num_mc_samples, 1, 1))

        output = self._model(z_particles)
        preds = output[:, :, :self._state_dimen]

        if self._predict_std:
            pred_log_stds = output[:, :, self._state_dimen:]
            preds_with_noise = preds + pred_log_stds.exp() * torch.randn_like(preds)
            return preds_with_noise.mean(dim=0), preds_with_noise.var(dim=0)
        else:
            return preds.mean(dim=0), preds.var(dim=0)

    def _update_model(self, x_train: Tensor, y_train: Tensor) -> None:
        # Nothing to do. We do not store the training data, just incorporate it in the model in _train_model().
        pass

    def _train_model(self, x_train: Tensor, y_train: Tensor) -> None:
        if self._reinitialize_on_train:
            # Construct an entirely new model to ensure all parameters are reinitialized correctly.
            self._model = self._model_constructor()

        self._model.train()

        optimizer = torch.optim.Adam(p for p in self._model.parameters() if p.requires_grad)

        # The size of y_train may be [N], but we require [N x n].
        if y_train.dim() == 1:
            y_train = y_train.unsqueeze(1)

        # TODO: should we reset the weights at the start of each training?
        print(f'Training BNN on {x_train.size(0)} data points for {self._training_iterations} iterations...')
        losses = []
        for i in range(self._training_iterations):
            optimizer.zero_grad()

            output = self._model(x_train, resample=True)
            pred_means = output[:, :self._state_dimen]
            pred_log_stds = output[:, self._state_dimen:] if self._predict_std else None

            loss = self._loss_function(y_train, pred_means, pred_log_stds)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        print(f'Training complete. Final losses: {losses[-4:]}')

        self._model.eval()

    def _fixed_dropout_loss(self, targets, pred_means, pred_log_stds):
        if pred_log_stds is not None:
            raise ValueError('Predicting aleatoric uncertainty is not supported for fixed dropout.')

        return F.mse_loss(pred_means, targets) + 1e-2 * self._model.regularization()

    def _concrete_dropout_loss(self, targets, pred_means, pred_log_stds):
        return (-self._gaussian_log_likelihood(targets, pred_means,
                                               pred_log_stds) + 1e-2 * self._model.regularization()).mean()

    @staticmethod
    def _gaussian_log_likelihood(targets, pred_means, pred_log_stds):
        """Taken from https://github.com/anassinator/bnn/blob/master/examples/sin_x.ipynb"""
        deltas = pred_means - targets

        if pred_log_stds is not None:
            pred_stds = pred_log_stds.exp()
            # TODO: does the np.log below cause a speed problem?
            return - ((deltas / pred_stds) ** 2).sum(-1) * 0.5 - pred_stds.log().sum(-1) - np.log(2 * math.pi) * 0.5
        else:
            return - (deltas ** 2).sum(-1) * 0.5

    def collect_metrics(self) -> Dict[str, Any]:
        dropout_ps = self._get_dropout_probabilities()
        return dropout_ps

    def _get_dropout_probabilities(self) -> Dict[str, float]:
        ps = dict()
        for i, layer in enumerate(self._model.children()):
            if isinstance(layer, (CDropout, BDropout)):
                # p is the inverse of the dropout rate.
                ps[f'dropout_p_layer_{i}'] = 1 - layer.p.item()
        return ps
