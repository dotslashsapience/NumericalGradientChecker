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

def _dictionary_to_vector(parameters_dict):
    """Unpack all parameter values from a dictionary into a single 1D vector."""
    all_vectors = []
    # iterate through param dict and unpack each matrix to apply NumPy ravel() to convert to 1D array
    for param_name in sorted(parameters_dict.keys()):
        # this will convert each matrix into individual vectors and append those vectors to all_vectors
        all_vectors.append(parameters_dict[param_name].ravel())

    # concatenate all vectors to create a single long vector.
    vector = np.concatenate(all_vectors)
    return vector

def compare_gradients(analytical_grads, numerical_grads, tolerance=1e-7):
    """Compare the analytical gradients from a backpropagation algorithm to the numerical gradients.

    Prints a PASS/FAIL message.

    Args:
        analytical_grads: dict of grads derived by Backprop algorithm.
        numerical_grads: dict of grads derived by numerical gradient checker.
        tolerance: float, the threshold for passing"""
    # convert both gradient dicts to vector.
    ana_grads_vec = _dictionary_to_vector(analytical_grads)
    num_grads_vec = _dictionary_to_vector(numerical_grads)
    # compute L2 norm of the difference between vectors for the numerator
    numerator = np.linalg.norm(ana_grads_vec - num_grads_vec)
    #compute the L2 norm of both vectors for the denominator
    denom_ana = np.linalg.norm(ana_grads_vec)
    denom_num = np.linalg.norm(num_grads_vec)
    #get the denominator
    denominator = denom_ana + denom_num
    relative_error = numerator / denominator

    if relative_error < tolerance:
        print(f"Gradient check PASSED")
        print(f"Relative Error: {relative_error:.2e}")
        print(f"Tolerance:      {tolerance:.2e}")
    else:
        print(f"Gradient check FAILED")
        print(f"Relative Error: {relative_error:.2e}")
        print(f"Tolerance:      {tolerance:.2e}")

    return relative_error


