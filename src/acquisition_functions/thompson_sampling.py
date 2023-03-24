#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

from __future__ import annotations

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples
from copy import copy

from src.models.variational_preferential_gp import VariationalPreferentialGP
from src.utils import optimize_acqf_and_get_suggested_query


def gen_thompson_sampling_query(model, batch_size, bounds, num_restarts, raw_samples):
    query = []
    for _ in range(batch_size):
        model_rff_sample = get_pairwise_gp_rff_sample(model=model, n_samples=1)
        acquisition_function = PosteriorMean(
            model=model_rff_sample
        )  # Approximate sample from the GP posterior
        new_x = optimize_acqf_and_get_suggested_query(
            acq_func=acquisition_function,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            batch_size=1,  # Batching is not supported by RFFs-based sample constructor
            batch_limit=1,
            init_batch_limit=1,
        )
        query.append(new_x.clone())

    query = torch.cat(query, dim=-2)
    query = query.unsqueeze(0)
    return query


def get_pairwise_gp_rff_sample(model, n_samples):
    model = model.eval()
    # Adapt the model so that it has training inputs and outputs. This is required to draw RFFs-based samples
    if isinstance(model, VariationalPreferentialGP):
        adapted_model = copy(model)
        queries_items = adapted_model.train_inputs[0]
        sample_at_queries_items = adapted_model.posterior(queries_items).sample()
        sample_at_queries_items = sample_at_queries_items.view(
            (queries_items.shape[0],)
        )
        adapted_model.train_targets = sample_at_queries_items
    # Draw RFFs-based (approximate) GP sample
    gp_samples = get_gp_samples(
        model=adapted_model,
        num_outputs=1,
        n_samples=n_samples,
        num_rff_features=1000,
    )

    return gp_samples
