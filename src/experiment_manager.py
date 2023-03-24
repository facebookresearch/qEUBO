#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional

from src.pbo_trial import pbo_trial


def experiment_manager(
    problem: str,
    obj_func: Callable,
    input_dim: int,
    noise_type: str,
    noise_level: float,
    algo: str,
    num_alternatives: int,
    num_init_queries: int,
    num_algo_queries: int,
    first_trial: int,
    last_trial: int,
    restart: bool,
    model_type: str = "variational_preferential_gp",
    add_baseline_point: bool = False,
    ignore_failures: bool = False,
    algo_params: Optional[Dict] = None,
) -> None:
    r"""
    Args:
        problem: Problem ID
        obj_func: The decision-maker's latent utility function
        input_dim: Input dimension
        noise_type: Type of noise in the decision-maker's responses (options: logit)
        noise_level: Noise level
        algo: Acquisition function
        num_alternatives: Number of alternatives in each query
        num_init_queries: Number of intial queries (chosen uniformly at random)
        num_algo_queries: Number of queries to be chosen using the acquisition function
        first_trial: First trial to be ran (This function runs all trials between first_trial and last_trial sequentially)
        last_trial: Last trial to be ran
        restart: If true, it will try to restart the experiment from available data
        model_type: Type of model (see utils.py for options)
        add_baseline_point: If true, it adds an initial set of queries against a "baseline point" (baseline point is hardcoded in utils.py)
    """
    for trial in range(first_trial, last_trial + 1):
        pbo_trial(
            problem=problem,
            obj_func=obj_func,
            input_dim=input_dim,
            noise_type=noise_type,
            noise_level=noise_level,
            algo=algo,
            algo_params=algo_params,
            num_alternatives=num_alternatives,
            num_init_queries=num_init_queries,
            num_algo_queries=num_algo_queries,
            trial=trial,
            restart=restart,
            model_type=model_type,
            add_baseline_point=add_baseline_point,
            ignore_failures=ignore_failures,
        )
