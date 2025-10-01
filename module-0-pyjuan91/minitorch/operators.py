"""Collection of the core mathematical operators used throughout the code base.

This module implements fundamental mathematical operations that serve as building blocks
for neural network computations in MiniTorch.

NOTE: The `task0_1` tests will not fully pass until you complete `task0_3`.
Some tests depend on higher-order functions implemented in the later task.
"""

import math
from typing import Callable, Iterable, List

# =============================================================================
# Task 0.1: Mathematical Operators
# =============================================================================

# """
# Implementation of elementary mathematical functions.

# FUNCTIONS TO IMPLEMENT:
#     Basic Operations:
#     - mul(x, y)     → Multiply two numbers
#     - id(x)         → Return input unchanged (identity function)
#     - add(x, y)     → Add two numbers
#     - neg(x)        → Negate a number

#     Comparison Operations:
#     - lt(x, y)      → Check if x < y
#     - eq(x, y)      → Check if x == y
#     - max(x, y)     → Return the larger of two numbers
#     - is_close(x, y) → Check if two numbers are approximately equal

#     Activation Functions:
#     - sigmoid(x)    → Apply sigmoid activation: 1/(1 + e^(-x))
#     - relu(x)       → Apply ReLU activation: max(0, x)

#     Mathematical Functions:
#     - log(x)        → Natural logarithm
#     - exp(x)        → Exponential function
#     - inv(x)        → Reciprocal (1/x)

#     Derivative Functions (for backpropagation):
#     - log_back(x, d)  → Derivative of log: d/x
#     - inv_back(x, d)  → Derivative of reciprocal: -d/(x²)
#     - relu_back(x, d) → Derivative of ReLU: d if x>0, else 0

# IMPORTANT IMPLEMENTATION NOTES:

# Numerically Stable Sigmoid:
#    To avoid numerical overflow, use different formulations based on input sign:

#    For x ≥ 0:  sigmoid(x) = 1/(1 + exp(-x))
#    For x < 0:  sigmoid(x) = exp(x)/(1 + exp(x))

#    Why? This prevents computing exp(large_positive_number) which causes overflow.

# is_close Function:
#    Use tolerance: |x - y| < 1e-2
#    This handles floating-point precision issues in comparisons.

# Derivative Functions (Backpropagation):
#    These compute: derivative_of_function(x) × upstream_gradient

#    - log_back(x, d):  d/dx[log(x)] = 1/x  →  return d/x
#    - inv_back(x, d):  d/dx[1/x] = -1/x**2   →  return -d/(x**2)
#    - relu_back(x, d): d/dx[relu(x)] = 1 if x>0 else 0  →  return d if x>0 else 0
# """


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: Product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Identity function that returns the input unchanged.

    Args:
        x (float): Input number.

    Returns:
        float: The same input number.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: Sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
        x (float): Input number.

    Returns:
        float: Negation of x.

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Check if x is less than y.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        bool: True if x < y, else False.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if x is equal to y.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        bool: True if x == y, else False.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Return the larger of two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The larger of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are approximately equal within a tolerance.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        bool: True if |x - y| < 1e-2, else False.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function.

    Args:
        x (float): Input number.

    Returns:
        float: Sigmoid of x.

    """
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)


def relu(x: float) -> float:
    """ReLU activation function.

    Args:
        x (float): Input number.

    Returns:
        float: ReLU of x.

    """
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Natural logarithm function.

    Args:
        x (float): Input number (must be > 0).

    Returns:
        float: Natural logarithm of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function.

    Args:
        x (float): Input number.

    Returns:
        float: e raised to the power of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Reciprocal function.

    Args:
        x (float): Input number (must be != 0).

    Returns:
        float: Reciprocal of x.

    """
    return 1 / x


def log_back(x: float, d: float) -> float:
    """Derivative of log function for backpropagation.

    Args:
        x (float): Input number (must be > 0).
        d (float): Upstream gradient.

    Returns:
        float: Gradient of log at x multiplied by upstream gradient d.

    """
    return d / x


def inv_back(x: float, d: float) -> float:
    """Derivative of reciprocal function for backpropagation.

    Args:
        x (float): Input number (must be != 0).
        d (float): Upstream gradient.

    Returns:
        float: Gradient of inv at x multiplied by upstream gradient d.

    """
    return -d / (x**2)


def relu_back(x: float, d: float) -> float:
    """Derivative of ReLU function for backpropagation.

    Args:
        x (float): Input number.
        d (float): Upstream gradient.

    Returns:
        float: Gradient of ReLU at x multiplied by upstream gradient d.

    """
    return d if x > 0 else 0.0


# =============================================================================
# Task 0.3: Higher-Order Functions
# =============================================================================

# """
# Implementation of functional programming concepts using higher-order functions.

# These functions work with other functions as arguments, enabling powerful
# abstractions for list operations.

# CORE HIGHER-ORDER FUNCTIONS TO IMPLEMENT:

#     map(fn, iterable):
#         Apply function `fn` to each element of `iterable`
#         Example: map(lambda x: x*2, [1,2,3]) → [2,4,6]

#     zipWith(fn, list1, list2):
#         Combine corresponding elements from two lists using function `fn`
#         Example: zipWith(add, [1,2,3], [4,5,6]) → [5,7,9]

#     reduce(fn, iterable, initial_value):
#         Reduce iterable to single value by repeatedly applying `fn`
#         Example: reduce(add, [1,2,3,4], 0) → 10

# FUNCTIONS TO BUILD USING THE ABOVE:

#     negList(lst):
#         Negate all elements in a list
#         Implementation hint: Use map with the neg function

#     addLists(lst1, lst2):
#         Add corresponding elements from two lists
#         Implementation hint: Use zipWith with the add function

#     sum(lst):
#         Sum all elements in a list
#         Implementation hint: Use reduce with add function and initial value 0

#     prod(lst):
#         Calculate product of all elements in a list
#         Implementation hint: Use reduce with mul function and initial value 1
# """


def map(fn: Callable[[float], float], iterable: Iterable[float]) -> Iterable[float]:
    """Apply function `fn` to each element of `iterable`.

    Args:
        fn (Callable[[float], float]): Function to apply.
        iterable (Iterable[float]): Iterable of numbers.

    Returns:
        Iterable[float]: New iterable with `fn` applied to each element.

    """
    return [fn(x) for x in iterable]


def zipWith(
    fn: Callable[[float, float], float], lst1: List[float], lst2: List[float]
) -> List[float]:
    """Combine corresponding elements from two lists using function `fn`.

    Args:
        fn (Callable[[float, float], float]): Function to combine elements.
        lst1 (List[float]): First list of numbers.
        lst2 (List[float]): Second list of numbers.

    Returns:
        List[float]: New list with `fn` applied to corresponding elements.

    """
    return [fn(x, y) for x, y in zip(lst1, lst2)]


def reduce(
    fn: Callable[[float, float], float], iterable: Iterable[float], initial_value: float
) -> float:
    """Reduce iterable to single value by repeatedly applying `fn`.

    Args:
        fn (Callable[[float, float], float]): Function to apply.
        iterable (Iterable[float]): Iterable of numbers.
        initial_value (float): Initial value for reduction.

    Returns:
        float: Reduced single value.

    """
    result = initial_value
    for x in iterable:
        result = fn(result, x)
    return result


def negList(lst: List[float]) -> List[float]:
    """Negate all elements in a list.

    Args:
        lst (List[float]): List of numbers.

    Returns:
        List[float]: New list with all elements negated.

    """
    return list(map(neg, lst))


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Add corresponding elements from two lists.

    Args:
        lst1 (List[float]): First list of numbers.
        lst2 (List[float]): Second list of numbers.

    Returns:
        List[float]: New list with corresponding elements added.

    """
    return zipWith(add, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Sum all elements in a list.

    Args:
        lst (List[float]): List of numbers.

    Returns:
        float: Sum of all elements.

    """
    return reduce(add, lst, 0.0)


def prod(lst: List[float]) -> float:
    """Calculate product of all elements in a list.

    Args:
        lst (List[float]): List of numbers.

    Returns:
        float: Product of all elements.

    """
    return reduce(mul, lst, 1.0)
