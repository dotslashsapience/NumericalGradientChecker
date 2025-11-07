# Elementary Numerical Gradient Checker
___
## Purpose
This numerical gradient checker is a very basic introduction
to gradient checking. Gradient Checking is an integral part
of Neural Network Development workflows. Neural networks rely
on a properly implemented Backpropagation algorithm to 
determine the gradient needed to minimize error and train
models. An improperly implemented Backprop algorithm may still
compile/run without generating errors but may never successfully
train the model.
<br><br>
I built this as part of a series of projects on my self-learning path
to building Neural Networks of my own. This is the first iteration of
my numerical gradient checker and is useful for getting a grasp on
the fundamentals. If you found this project as a beginner on your
own learning path, I recommend you study this before moving on to a
more robust gradient checker. If I don't have a second implementation up
I will soon.
___
## Limitations and Usage
This implementation should not be used to test production grade
systems or large neural networks. This is a very basic implementation
intended to familiarize myself and any others that may wish to
learn how to implement Neural Networks of their own.
### Time Complexity: O(n)
The time complexity of this implentation is O(n) due to the fact
that it tests every single parameter in the gradient. This is why
it should not be used for large Neural Networks.
___
## Architecture
This is a very simplistic setup that includes 3 functions. <br><br>
### 1. `compute_numerical_gradient()`
The first function `compute_numerical_gradient()`, a `forward_prop_func`
`X` for **input data** and `y` for **labels** (assumes a classification
problem) handles the numerical computation of the gradients using the 
**Central Difference Formula:**
$$
f'(x) \approx \frac{f(x + \epsilon) - f(x-\epsilon)}{2\epsilon}
$$
It takes in a `dict` of `parameters` which should be a series of
matrices associated with a label e.g. `W1`. A `dict`, `numerical_grads` is generated with
matching names for keys and values generated are matrices of equal dimensions
to their respective counterparts in `parameters`. All values in the matrices are
initialized to `0.0`. 
<br><br>
It then iterates through each of the values in each of the matrices
in `parameters`, storing the original value, adding $\epsilon$ and 
conducting a `forward_prop_func` pass to get the cost value. repeating
the process with $x-\epsilon$ and then divides the difference by $2\epsilon$
to get the estimated gradient. The gradient is then added to the associated
index in the `numerical_grads` dict.
<br><br>
Once complete, the function returns the `numerical_grads` dict.
___
### 2. `compare_gradients()`
The second function `compare_gradients()` takes in the `analytical_grads`
dict which is derived from your backprop algorithm, the `numerical_grads`
dict derived from `compute_numerical_gradient()` and a `tolerance` which
defaults to $1\cdot10^{-7}$.<br><br>
The formula used to determine error is:
$$
\text{Relative Error} = \frac{\|\text{analytic gradient} - \text{numeric gradient} \|}{\|\text{analytic gradient}\|+ \|\text{numeric gradient}  \|}
$$
It then extracts the matrices from the dicts and converts each dict into
individual vectors. The $L_2$ norm is used and the $\text{Relative Error}$
is calculated. The function then prints a pass/fail report.
___
### 3. `_dictionary_to_vector()`
This is a simple helper function that takes int a `parameters_dict`
and returns a Vector of the parameter values.

