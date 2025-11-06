import numpy as np
from typing import Sequence

def compute_numerical_gradient(parameters, forward_prop_func, X, y, epsilon=1e-5):
    """Calculates the numerical gradient for all parameters in the 'parameters' dictionary using
    the central difference formula.
    This is a "black-box" check that perturbs each parameter individually and observes the change in the final cost.

    Arguments:
        parameters -- dict, example: {"W1": W1, "b1": b1, "W2", W2...}
        forward_prop_func -- a function that takes (X, y, parameters) and returns the cost.
        X -- input data
        y -- labels
        epsilon -- the small perturbation value.

    Returns:
        numerical_grads -- dict, same structure as 'parameters', containing the numerical gradients for each param."""
    numerical_grads = {}
    for param_name, param_values in parameters.items():
        numerical_grads[param_name] = np.zeros_like(param_values)

    for param_name, param_values in parameters.items():
        num_grad_for_param = numerical_grads[param_name]
        it = np.nditer(param_values, flags=['multi_index'], op_flags=[['readwrite']])

        while not it.finished:
            idx = it.multi_index
            original_value = param_values[idx]
            param_values[idx] = original_value + epsilon

            cost_plus, _ = forward_prop_func(X, y, parameters)

