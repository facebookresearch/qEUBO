#!/usr/bin/env python3

import os
import sys
import torch
from botorch.settings import debug
from botorch.test_functions.synthetic import Hartmann
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager


# Objective function
input_dim = 6


def obj_func(X: Tensor) -> Tensor:
    hartmann = Hartmann()
    objective_X = -hartmann.evaluate_true(X)
    return objective_X


# Algos
# algo = "random"
# algo = "analytic_eubo"
algo = "qeubo"
# algo = "qei"
# algo = "qnei"
# algo = "qts"
# algo = "mpes"

# Noise level
noise_type = "logit"
noise_level_id = 2

if noise_type == "logit":
    noise_levels = [0.0673, 0.1619, 0.3242]

noise_level = noise_levels[noise_level_id - 1]

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="hartmann",
    obj_func=obj_func,
    input_dim=input_dim,
    noise_type=noise_type,
    noise_level=noise_level,
    algo=algo,
    num_alternatives=2,
    num_init_queries=4 * input_dim,
    num_algo_queries=150,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
