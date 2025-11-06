import numpy as np

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
    # initialize dict
    numerical_grads = {}
    # add entries to dict to match params
    for param_name, param_values in parameters.items():
        # initialize empty matrix values for keys and match dimensions.
        numerical_grads[param_name] = np.zeros_like(param_values)
    # iterate through parameters dictionary
    for param_name, param_values in parameters.items():
        # initiate an empty matrix for numerical_grads that matches the dimensions of respective parameters matrix.
        num_grad_for_param = numerical_grads[param_name]
        # get n-dimensional iterator for each matrix
        it = np.nditer(param_values, flags=['multi_index'], op_flags=[['readwrite']])

        while not it.finished:
            # iterate through each matrix.
            # iterator changes with each iteration of while loop.
            idx = it.multi_index
            # store original value
            original_value = param_values[idx]
            # add epsilon and update parameter in matrix
            param_values[idx] = original_value + epsilon
            # run a full forward pass with the updates parameter to get cost.
            cost_plus, _ = forward_prop_func(X, y, parameters)
            # subtract eps and update parameter in matrix
            param_values[idx] = original_value - epsilon
            # run forward pass with the cost minus epsilon
            cost_minus, _ = forward_prop_func(X, y, parameters)
            # Calculate the numerical gradient and assign it to the num_grad_for_param
            num_grad_for_param[idx] = (cost_plus - cost_minus) / (2 * epsilon)
            # reassign the param value to the original value
            param_values[idx] = original_value
            # increment iterator
            it.iternext()

    return numerical_grads

