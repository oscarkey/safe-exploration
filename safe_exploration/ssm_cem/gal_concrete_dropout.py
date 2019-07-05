"""Concrete dropout from paper "Concrete Dropout"; Gal, Hron, Kendall.

Implementation taken from:
https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-pytorch.ipynb
"""
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch import nn, optim, Tensor

from .ssm_cem import CemSSM
from ..ssm_pytorch import utilities
from ..utils import get_device


class _ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(_ConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)

        out = layer(self._concrete_dropout(x, p))

        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        input_dimensionality = x[0].numel()  # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(unif_noise + eps) - torch.log(
            1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x

    @property
    def p(self) -> float:
        return torch.sigmoid(self.p_logit).item()


class _Model(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: List[int], weight_regularizer,
                 dropout_regularizer):
        super(_Model, self).__init__()

        assert len(hidden_features) == 2, f'We only support networks with two hidden layers, got {hidden_features}'

        self.linear1 = nn.Linear(in_features, hidden_features[0])
        self.linear2 = nn.Linear(hidden_features[0], hidden_features[1])

        self.linear3_mu = nn.Linear(hidden_features[1], out_features)
        self.linear3_logvar = nn.Linear(hidden_features[1], out_features)

        self.conc_drop1 = _ConcreteDropout(weight_regularizer=weight_regularizer,
                                           dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = _ConcreteDropout(weight_regularizer=weight_regularizer,
                                           dropout_regularizer=dropout_regularizer)
        self.conc_drop_mu = _ConcreteDropout(weight_regularizer=weight_regularizer,
                                             dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar = _ConcreteDropout(weight_regularizer=weight_regularizer,
                                                 dropout_regularizer=dropout_regularizer)

        self.relu = nn.ReLU()

    def forward(self, x):
        regularization = torch.empty(4, device=x.device)

        x1, regularization[0] = self.conc_drop1(x, nn.Sequential(self.linear1, self.relu))
        x2, regularization[1] = self.conc_drop2(x1, nn.Sequential(self.linear2, self.relu))

        mean, regularization[2] = self.conc_drop_mu(x2, self.linear3_mu)
        log_var, regularization[3] = self.conc_drop_logvar(x2, self.linear3_logvar)

        return mean, log_var, regularization.sum()

    def get_dropout_probabilities(self) -> Dict[str, float]:
        ps = dict()
        ps['conc_drop1'] = self.conc_drop1.p
        ps['conc_drop2'] = self.conc_drop2.p
        ps['conc_drop_mu'] = self.conc_drop_mu.p
        ps['conc_drop_logvar'] = self.conc_drop_logvar.p
        return ps


def _heteroscedastic_loss(true, mean, log_var):
    precision = torch.exp(-log_var)
    return torch.mean(torch.sum(precision * (true - mean) ** 2 + log_var, 1), 0)


_batch_size = 20


class GalConcreteDropoutSSM(CemSSM):
    """A BNN state space model, approximated using concrete mc dropout.

    Uses the implementation from the paper "Concrete Dropout"; Gal, Hron, Kendall at
    https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-pytorch.ipynb
    """

    def __init__(self, conf, state_dimen: int, action_dimen: int):
        super().__init__(state_dimen, action_dimen)

        assert conf.mc_dropout_on_input is True
        assert conf.mc_dropout_type == 'concrete'
        assert conf.mc_dropout_predict_std is True
        # TODO: use parameter mc_dropout_concrete_initial_probability

        self._num_samples = conf.mc_dropout_num_samples
        self._training_iterations = conf.mc_dropout_training_iterations
        self._hidden_features = conf.mc_dropout_hidden_features
        self._length_scale = conf.mc_dropout_lengthscale
        self._device = get_device(conf)

        self._model_constructor = self._get_model_constructor(state_dimen, action_dimen)
        # Use any value for weight_regularizer and dropout_regularizer as we don't yet have any data.
        self._model = self._model_constructor(weight_regularizer=1.0, dropout_regularizer=1.0)

    def _get_model_constructor(self, state_dimen: int, action_dimen: int):
        in_features = state_dimen + action_dimen
        out_features = state_dimen

        def constructor(weight_regularizer, dropout_regularizer):
            model = _Model(in_features, out_features, self._hidden_features, weight_regularizer, dropout_regularizer)
            model.to(self._device)
            return model

        return constructor

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

    def predict_raw(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        self._model.eval()
        mc_samples = [self._model(z) for _ in range(self._num_samples)]

        means = torch.stack([output[0] for output in mc_samples])
        log_vars = torch.stack([output[1] for output in mc_samples])

        pred_mean = means.mean(0)

        epistemic_uncertainty = means.var(0)
        aleatoric_uncertainty = log_vars.mean(0).exp()
        # TODO: Include aleatoric uncertainty.
        total_var = epistemic_uncertainty

        return pred_mean, total_var

    def _update_model(self, x_train: Tensor, y_train: Tensor) -> None:
        # Nothing to do. We do not store the training data, just incorporate it in the model in _train_model().
        pass

    def _train_model(self, x_train: Tensor, y_train: Tensor) -> None:
        # We require that the shape of the training data is [N x 1], not just [N].
        if y_train.dim() == 1:
            y_train = y_train.unsqueeze(1)

        N = x_train.shape[0]
        weight_regularizer = self._length_scale ** 2. / N
        dropout_regularizer = 2. / N
        model = self._model_constructor(weight_regularizer, dropout_regularizer)
        model.train()
        optimizer = optim.Adam(model.parameters())

        for i in range(self._training_iterations):
            old_batch = 0
            for batch in range(int(np.ceil(x_train.shape[0] / _batch_size))):
                batch = (batch + 1)
                x = x_train[old_batch: _batch_size * batch]
                y = y_train[old_batch: _batch_size * batch]

                mean, log_var, regularization = model(x)

                loss = _heteroscedastic_loss(y, mean, log_var) + regularization

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % 500 == 0:
                print(f'Training epoch {i}/{self._training_iterations - 1}, loss={loss.item()}')

        self._model = model

    def collect_metrics(self) -> Dict[str, float]:
        dropout_ps = self._model.get_dropout_probabilities()
        metrics = {'dropout_p_' + k: v for k, v in dropout_ps.items()}
        return metrics
