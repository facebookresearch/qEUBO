#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)
from torch import Tensor

from src.models.likelihoods.preferential_softmax_likelihood import (
    PreferentialSoftmaxLikelihood,
)


class VariationalPreferentialGP(GPyTorchModel, ApproximateGP):
    def __init__(
        self,
        queries: Tensor,
        responses: Tensor,
        use_withening: bool = True,
        covar_module: Optional[Kernel] = None,
    ) -> None:
        r"""
        Args:
            queries: A `n x q x d` tensor of training inputs. Each of the `n` queries is constituted
                by `q` `d`-dimensional decision vectors.
            responses: A `n x 1` tensor of training outputs. Each of the `n` responses is an integer
                between 0 and `q-1` indicating the decision vector selected by the user.
            use_withening: If true, use withening to enhance variational inference.
            covar_module: The module computing the covariance matrix.
        """
        self.queries = queries
        self.responses = responses
        self.input_dim = queries.shape[-1]
        self.q = queries.shape[-2]
        self.num_data = queries.shape[-3]
        train_x = queries.reshape(
            queries.shape[0] * queries.shape[1], queries.shape[2]
        )  # Reshape queries in the form of "standard training inputs"
        train_y = responses.squeeze(-1)  # Squeeze out output dimension
        bounds = torch.tensor(
            [[0, 1] for _ in range(self.input_dim)], dtype=torch.double
        ).T  # This assumes the input space has been normalized beforehand
        # Construct variational distribution and strategy
        if use_withening:
            inducing_points = draw_sobol_samples(
                bounds=bounds,
                n=2 * self.input_dim,
                q=1,
                seed=0,
            ).squeeze(1)
            inducing_points = torch.cat([inducing_points, train_x], dim=0)
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2)
            )
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            )
        else:
            inducing_points = train_x
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2)
            )
            variational_strategy = UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            )
        super().__init__(variational_strategy)
        self.likelihood = PreferentialSoftmaxLikelihood(num_alternatives=self.q)
        self.mean_module = ConstantMean()
        scales = bounds[1, :] - bounds[0, :]

        if covar_module is None:
            self.covar_module = ScaleKernel(
                RBFKernel(
                    ard_num_dims=self.input_dim,
                    lengthscale_prior=GammaPrior(3.0, 6.0 / scales),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        else:
            self.covar_module = covar_module
        self._num_outputs = 1
        self.train_inputs = (train_x,)
        self.train_targets = train_y

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)
        return MultivariateNormal(mean_X, covar_X)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1
