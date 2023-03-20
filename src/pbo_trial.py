#!/usr/bin/env python3

from typing import Callable, Dict, Optional

import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import (
    PosteriorMean,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.models.model import Model
from botorch.sampling import SobolQMCNormalSampler
from torch import Tensor

from src.acquisition_functions.eubo import (
    ExpectedUtilityOfBestOption,
    qExpectedUtilityOfBestOption,
)
from src.acquisition_functions.mpes import MultinomialPredictiveEntropySearch
from src.acquisition_functions.thompson_sampling import gen_thompson_sampling_query
from src.utils import (
    fit_model,
    generate_initial_data,
    generate_random_queries,
    get_obj_vals,
    generate_responses,
    optimize_acqf_and_get_suggested_query,
)

# See experiment_manager.py for parameters
def pbo_trial(
    problem: str,
    obj_func: Callable,
    input_dim: int,
    noise_type: str,
    noise_level: float,
    algo: str,
    num_alternatives: int,
    num_init_queries: int,
    num_algo_queries: int,
    trial: int,
    restart: bool,
    model_type: str,
    add_baseline_point: bool,
    ignore_failures: bool,
    algo_params: Optional[Dict] = None,
) -> None:

    algo_id = algo + "_" + str(num_alternatives)  # Append q to algo ID

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = (
        project_path + "/experiments/results/" + problem + "/" + algo_id + "/"
    )

    if restart:
        # Check if training data is already available
        try:
            # Current available evaluations
            queries = np.loadtxt(
                results_folder + "queries/queries_" + str(trial) + ".txt"
            )
            queries = queries.reshape(
                queries.shape[0],
                num_alternatives,
                int(queries.shape[1] / num_alternatives),
            )
            queries = torch.tensor(queries)
            obj_vals = torch.tensor(
                np.loadtxt(results_folder + "obj_vals/obj_vals_" + str(trial) + ".txt")
            )
            responses = torch.tensor(
                np.loadtxt(
                    results_folder + "responses/responses_" + str(trial) + ".txt"
                )
            )
            # Historical maximum objective values within queries
            max_obj_vals_within_queries = list(
                np.loadtxt(
                    results_folder
                    + "max_obj_vals_within_queries_"
                    + str(trial)
                    + ".txt"
                )
            )
            # Historical objective values at the maximum of the posterior mean
            obj_vals_at_max_post_mean = list(
                np.loadtxt(
                    results_folder + "obj_vals_at_max_post_mean_" + str(trial) + ".txt"
                )
            )
            # Historical acquisition runtimes
            runtimes = list(
                np.atleast_1d(
                    np.loadtxt(
                        results_folder + "runtimes/runtimes_" + str(trial) + ".txt"
                    )
                )
            )

            # Fit GP model
            t0 = time.time()
            model = fit_model(
                queries,
                responses,
                model_type=model_type,
                likelihood=noise_type,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            iteration = len(max_obj_vals_within_queries) - 1
            print("Restarting experiment from available data.")

        except:
            # Initial data
            queries, obj_vals, responses = generate_initial_data(
                num_queries=num_init_queries,
                num_alternatives=num_alternatives,
                input_dim=input_dim,
                obj_func=obj_func,
                noise_type=noise_type,
                noise_level=noise_level,
                add_baseline_point=add_baseline_point,
                seed=trial,
            )

            # Fit GP model
            t0 = time.time()
            model = fit_model(
                queries,
                responses,
                model_type=model_type,
                likelihood=noise_type,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            # Historical objective values at the maximum of the posterior mean
            obj_val_at_max_post_mean = compute_obj_val_at_max_post_mean(
                obj_func, model, input_dim
            )
            obj_vals_at_max_post_mean = [obj_val_at_max_post_mean]

            # Historical maximum objective values within queries and runtimes
            max_obj_val_within_queries = obj_vals.max().item()
            max_obj_vals_within_queries = [max_obj_val_within_queries]

            # Historical acquisition runtimes
            runtimes = []

            iteration = 0
    else:
        # Initial data
        queries, obj_vals, responses = generate_initial_data(
            num_queries=num_init_queries,
            num_alternatives=num_alternatives,
            input_dim=input_dim,
            obj_func=obj_func,
            noise_type=noise_type,
            noise_level=noise_level,
            add_baseline_point=add_baseline_point,
            seed=trial,
        )

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            queries,
            responses,
            model_type=model_type,
            likelihood=noise_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # Historical objective values at the maximum of the posterior mean
        obj_val_at_max_post_mean = compute_obj_val_at_max_post_mean(
            obj_func, model, input_dim
        )
        obj_vals_at_max_post_mean = [obj_val_at_max_post_mean]

        # Historical maximum objective values within queries and runtimes
        max_obj_val_within_queries = obj_vals.max().item()
        max_obj_vals_within_queries = [max_obj_val_within_queries]

        # Historical acquisition runtimes
        runtimes = []

        iteration = 0

    while iteration < num_algo_queries:
        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + algo_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # New suggested query
        t0 = time.time()
        new_query = get_new_suggested_query(
            algo=algo,
            model=model,
            num_alternatives=num_alternatives,
            input_dim=input_dim,
            algo_params=algo_params,
            noise_level=noise_level,
            model_type=model_type,
        )
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # Get response at new query
        new_obj_vals = get_obj_vals(new_query, obj_func)
        new_response = generate_responses(
            new_obj_vals, noise_type=noise_type, noise_level=noise_level
        )

        # Update training data
        queries = torch.cat((queries, new_query))
        obj_vals = torch.cat([obj_vals, new_obj_vals], 0)
        responses = torch.cat((responses, new_response))

        # Fit GP model
        t0 = time.time()
        model = fit_model(
            queries,
            responses,
            model_type=model_type,
            likelihood=noise_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # Append current objective value at the maximum of the posterior mean
        obj_val_at_max_post_mean = compute_obj_val_at_max_post_mean(
            obj_func, model, input_dim
        )
        obj_vals_at_max_post_mean.append(obj_val_at_max_post_mean)
        print(
            "Objective value at the maximum of the posterior mean: "
            + str(obj_val_at_max_post_mean)
        )

        # Append current max objective val within queries
        max_obj_val_within_queries = obj_vals.max().item()
        max_obj_vals_within_queries.append(max_obj_val_within_queries)
        print("Max objecive value within queries: " + str(max_obj_val_within_queries))

        # Save data
        try:
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            if not os.path.exists(results_folder + "queries/"):
                os.makedirs(results_folder + "queries/")
            if not os.path.exists(results_folder + "obj_vals/"):
                os.makedirs(results_folder + "obj_vals/")
            if not os.path.exists(results_folder + "responses/"):
                os.makedirs(results_folder + "responses/")
            if not os.path.exists(results_folder + "runtimes/"):
                os.makedirs(results_folder + "runtimes/")
        except:
            pass

        queries_reshaped = queries.numpy().reshape(queries.shape[0], -1)
        np.savetxt(
            results_folder + "queries/queries_" + str(trial) + ".txt", queries_reshaped
        )
        np.savetxt(
            results_folder + "obj_vals/obj_vals_" + str(trial) + ".txt",
            obj_vals.numpy(),
        )
        np.savetxt(
            results_folder + "responses/responses_" + str(trial) + ".txt",
            responses.numpy(),
        )
        np.savetxt(
            results_folder + "runtimes/runtimes_" + str(trial) + ".txt",
            np.atleast_1d(runtimes),
        )
        np.savetxt(
            results_folder + "obj_vals_at_max_post_mean_" + str(trial) + ".txt",
            np.atleast_1d(obj_vals_at_max_post_mean),
        )
        np.savetxt(
            results_folder + "max_obj_vals_within_queries_" + str(trial) + ".txt",
            np.atleast_1d(max_obj_vals_within_queries),
        )


# Computes new query to evaluate
def get_new_suggested_query(
    algo: str,
    model: Model,
    num_alternatives,
    input_dim: int,
    noise_level: float,
    model_type: str,
    algo_params: Optional[Dict] = None,
) -> Tensor:

    standard_bounds = torch.tensor(
        [[0.0] * input_dim, [1.0] * input_dim]
    )  # This assumes the input domain has been normalized beforehand
    num_restarts = input_dim * num_alternatives
    raw_samples = 30 * input_dim * num_alternatives
    batch_initial_conditions = None

    if algo == "random":
        return generate_random_queries(
            num_queries=1, num_alternatives=num_alternatives, input_dim=input_dim
        )
    elif algo == "analytic_eubo":
        acquisition_function = ExpectedUtilityOfBestOption(model=model)
    elif algo == "qeubo":
        sampler = SobolQMCNormalSampler(sample_shape=64)
        acquisition_function = qExpectedUtilityOfBestOption(
            model=model, sampler=sampler
        )
    elif algo == "mpes":
        standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
        acquisition_function = MultinomialPredictiveEntropySearch(
            model=model,
            bounds=standard_bounds,
        )
    elif algo == "qei":
        sampler = SobolQMCNormalSampler(sample_shape=64)
        if model_type == "variational_preferential_gp":
            X_baseline = model.queries.clone()
            X_baseline = X_baseline.view(
                (X_baseline.shape[0] * X_baseline.shape[1], X_baseline.shape[2])
            )
        posterior = model.posterior(X_baseline)
        mean = posterior.mean

        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=mean.max().item(),
            sampler=sampler,
        )
    elif algo == "qnei":
        sampler = SobolQMCNormalSampler(sample_shape=64)
        if model_type == "variational_preferential_gp":
            X_baseline = model.train_inputs[0]
        acquisition_function = qNoisyExpectedImprovement(
            model=model,
            X_baseline=X_baseline,
            sampler=sampler,
            prune_baseline=True,
        )
    elif algo == "qts":
        standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
        return gen_thompson_sampling_query(
            model, num_alternatives, standard_bounds, input_dim, 30 * input_dim
        )

    new_query = optimize_acqf_and_get_suggested_query(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=num_alternatives,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
    )

    new_query = new_query.unsqueeze(0)
    return new_query


# Computes the (true) objective value at the maximizer of the model's posterior mean function
def compute_obj_val_at_max_post_mean(
    obj_func: Callable,
    model: Model,
    input_dim: int,
) -> Tensor:

    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 6 * input_dim
    raw_samples = 180 * input_dim

    post_mean_func = PosteriorMean(model=model)
    max_post_mean_func = optimize_acqf_and_get_suggested_query(
        acq_func=post_mean_func,
        bounds=standard_bounds,
        batch_size=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    obj_val_at_max_post_mean_func = obj_func(max_post_mean_func).item()
    return obj_val_at_max_post_mean_func
