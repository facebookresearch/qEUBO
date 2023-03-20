#!/usr/bin/env python3

import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import Likelihood


class PreferentialSoftmaxLikelihood(Likelihood):
    r"""
    Implements the softmax likelihood used for GP-based preference learning.

    .. math::
        p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf f \right)

    :param int num_alternatives: Number of alternatives (i.e., q).
    """

    def __init__(self, num_alternatives):
        super().__init__()
        self.num_alternatives = num_alternatives
        self.noise = torch.tensor(1e-4)  # This is only used to draw RFFs-based
        # samples. We set it close to zero because we want noise-free samples
        self.sampler = SobolQMCNormalSampler(sample_shape=512)  # This allows for
        # SAA-based optimization of the ELBO

    def _draw_likelihood_samples(
        self, function_dist, *args, sample_shape=None, **kwargs
    ):
        function_samples = self.sampler(GPyTorchPosterior(function_dist)).squeeze(-1)
        return self.forward(function_samples, *args, **kwargs)

    def forward(self, function_samples, *params, **kwargs):
        function_samples = function_samples.reshape(
            function_samples.shape[:-1]
            + torch.Size(
                (
                    int(function_samples.shape[-1] / self.num_alternatives),
                    self.num_alternatives,
                )
            )
        )  # Reshape samples as if they came from a multi-output model (with `q` outputs)
        num_alternatives = function_samples.shape[-1]

        if num_alternatives != self.num_alternatives:
            raise RuntimeError("There should be %d points" % self.num_alternatives)

        res = base_distributions.Categorical(logits=function_samples)  # Passing the
        # function values as logits recovers the softmax likelihood
        return res
