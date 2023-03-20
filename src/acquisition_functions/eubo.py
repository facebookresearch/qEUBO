from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction, MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor


class ExpectedUtilityOfBestOption(AcquisitionFunction):
    r"""Analytic Expected Utility of Best Option (EUBO)"""

    def __init__(
        self,
        model: Model,
    ) -> None:
        r"""Analytic Expected Utility of Best Option (EUBO).

        Only supports the case of `q=1`. The model must be
        single-outcome.

        Args:
            model: A fitted single-outcome model.
        """
        super().__init__(model=model)
        self.standard_normal = torch.distributions.normal.Normal(
            torch.zeros(1),
            torch.ones(1),
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate EUBO on the candidate set X.
        Args:
            X: A `batch_shape x 2 x d`-dim Tensor.
        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        posterior = self.model(X)  # Calling directly instead of posterior here to
        # obtain the full covariance matrix
        mean = posterior.mean
        cov = posterior.covariance_matrix
        delta = mean[..., 0] - mean[..., 1]
        sigma = torch.sqrt(
            cov[..., 0, 0] + cov[..., 1, 1] - cov[..., 0, 1] - cov[..., 1, 0]
        )
        u = delta / sigma

        ucdf = self.standard_normal.cdf(u)
        updf = torch.exp(self.standard_normal.log_prob(u))
        acqf_val = sigma * (updf + u * ucdf)
        acqf_val += mean[..., 1]
        return acqf_val


class qExpectedUtilityOfBestOption(MCAcquisitionFunction):
    r"""Expected Utility of Best Option (qEUBO).

    This computes qEUBO by
    (1) sampling the joint posterior over q points
    (2) evaluating the maximum objective value accross the q points for each sample
    (3) averaging over the samples

    `qEUBO(X) = E[max Y], Y ~ f(X), where X = (x_1,...,x_q)`
    """

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_baseline: Optional[Tensor] = None,
    ) -> None:
        r"""MC-based Expected Utility of the Best Option (qEUBO).

        Args:
            model: A fitted model.
             sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            X_baseline:  A `m x d`-dim Tensor of `m` design points forced to be included
                in the query (in addition to the q points, so the query is constituted
                by q + m alternatives). Concatenated into X upon forward call. Copied and
                set to have no gradient. This is useful, for example, if we want to force
                one of the alternatives to be the point chosen by the decision-maker in
                the previous iteration.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            X_pending=X_baseline,
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qEUBO on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of qEUBO values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        posterior_X = self.model.posterior(X)
        Y_samples = self.sampler(posterior_X)
        util_val_samples = self.objective(Y_samples)
        best_util_val_samples = util_val_samples.max(dim=-1).values
        exp_best_util_val = best_util_val_samples.mean(dim=0)
        return exp_best_util_val
