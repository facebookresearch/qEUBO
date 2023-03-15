#!/usr/bin/env python3

import os
import sys
import torch
from botorch.settings import debug
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from experiments.problems import car_cab_obj_func, CarCabDesign
from src.experiment_manager import experiment_manager


# Objective function
input_dim = CarCabDesign().dim
bounds = torch.tensor(CarCabDesign()._bounds)


def obj_func(X: Tensor) -> Tensor:
    X_unnorm = (bounds[:, 1] - bounds[:, 0]) * X + bounds[:, 0]
    objective_X = car_cab_obj_func(X_unnorm)
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
noise_level = 0.0558

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="carcab",
    obj_func=obj_func,
    input_dim=input_dim,
    noise_type=noise_type,
    noise_level=noise_level,
    algo=algo,
    num_alternatives=2,
    num_init_queries=4 * input_dim,
    num_max_iter=150,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
