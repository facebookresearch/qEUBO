import torch

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from typing import Any


JITTER = 1e-2


class MultinomialPredictiveEntropySearch(MCAcquisitionFunction):
    r"""Multinomial Predictive Entropy Search (MPES)"""

    def __init__(
        self,
        model: Model,
        bounds: Tensor,
        n_samples: int = 2048,
        n_x_max_samples: int = 8,
        **kwargs: Any,
    ) -> None:
        self.bounds = bounds
        self.n_samples = n_samples
        self.n_x_max_samples = n_x_max_samples
        super().__init__(
            model=model,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([n_samples])),
        )
        self.softplus = torch.nn.Softplus()
        self.X_max = self.gen_X_max_cand_samples()

    def gen_X_max_cand_samples(self):
        with torch.no_grad():
            dim = self.bounds.shape[-1]
            n_x_max_subsamples = 1024
            X_max_candidates = torch.rand(self.n_x_max_samples, n_x_max_subsamples, dim)
            X_max_candidates = (
                X_max_candidates * (self.bounds[..., 1, :] - self.bounds[..., 0, :])
                + self.bounds[..., 0, :]
            )

            X_max_cand_post = self.model.posterior(X_max_candidates)
            X_max_cand_samples = X_max_cand_post.sample()
            sample_max_idx = X_max_cand_samples.squeeze(-1).squeeze(0).argmax(dim=-1)
            X_max = torch.index_select(
                input=X_max_candidates,
                dim=-2,
                index=sample_max_idx,
            )
            X_max = torch.diagonal(input=X_max, offset=0, dim1=-2, dim2=-3).T
            return X_max

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        batch_shape, q, dim = X.shape[:-2], X.shape[-2], X.shape[-1]
        X_max = self.X_max.expand(*batch_shape, self.n_x_max_samples, dim)

        # taking samples
        X_X_max = torch.cat((X, X_max), dim=-2)
        joint_post = self.model.posterior(X_X_max)
        joint_samples = self.sampler(joint_post).squeeze(-1).to(X)

        X_samples = joint_samples[..., :q]
        X_max_samples = joint_samples[..., q:]
        # P(x_star | D)
        x_max_indices = torch.arange(self.n_x_max_samples).expand(X_max_samples.shape)
        X_max_samples_id = (
            X_max_samples.argmax(dim=-1).unsqueeze(-1).expand(x_max_indices.shape)
        )
        X_max_count = torch.eq(x_max_indices, X_max_samples_id).to(X)
        X_max_count = self.softplus(X_max_count)
        X_max_prob = X_max_count.sum(dim=0)
        X_max_prob = X_max_prob / X_max_prob.sum(-1).unsqueeze(-1)
        # assert torch.allclose(X_max_prob.sum(-1), torch.tensor(1.).to(X_max_prob))

        # P(X, x_star | D)
        target_size = torch.Size(
            (self.n_samples, *batch_shape, self.n_x_max_samples, q)
        )
        X_max_count = X_max_count.unsqueeze(-1).expand(target_size)
        X_best_prob = torch.softmax(X_samples, dim=-1).unsqueeze(-2).expand(target_size)
        joint_prob = (X_max_count * X_best_prob).sum(dim=0)
        joint_prob = joint_prob / joint_prob.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
        # assert torch.allclose(joint_prob.sum(-1).sum(-1), torch.tensor(1.).to(X_max_prob))

        # P(X|D)
        X_prob = joint_prob.sum(dim=-2)

        # P(X | D, x_star) = P(X, x_star | D) / P(x_star | D)
        cond_prob = joint_prob / (X_max_prob.unsqueeze(-1))

        ratio = torch.log((cond_prob) / (X_prob.unsqueeze(-2)))
        mutual_info = (X_max_prob * (cond_prob * ratio).sum(-1)).sum(-1)

        return mutual_info
