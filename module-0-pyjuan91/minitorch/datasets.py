"""2D Classification Datasets for MiniTorch Visualization

This module provides various 2D point classification datasets used for testing
and visualizing machine learning models in MiniTorch.

PYRIGHT STYLE REQUIREMENTS:
To pass the type checking tests, you need to:

1. ADD TYPE ANNOTATIONS to all function parameters and return values
   Example: def make_pts(N: int) -> List[Tuple[float, float]]:

2. ADD DOCSTRINGS to all functions
   Use the triple-quote format with a brief description of what the function does

3.ENSURE ALL IMPORTS are properly typed
   The required imports are already provided at the top
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random 2D points within the unit square [0,1] x [0,1].

    Args:
        N (int): Number of points to generate.

    Returns:
        List[Tuple[float, float]]: List of N tuples representing the 2D points.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a simple dataset where points are classified based on the x-coordinate.
        Points with x-coordinate < 0.5 are labeled 1, others labeled 0.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A Graph dataclass instance containing the points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a diagonal dataset where points are classified based on the sum of their coordinates.
    Points with x1 + x2 < 0.5 are labeled 1, others labeled 0.

    Args:
        N (int): Number of points to generate.

    Returns:
            Graph: A Graph dataclass instance containing the points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a split dataset where points are classified based on the x-coordinate.
        Points with x-coordinate < 0.2 or > 0.8 are labeled 1, others labeled 0.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A Graph dataclass instance containing the points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate an XOR dataset where points are classified based on their position
        relative to the center of the unit square. Points in opposite quadrants are labeled 1,
        others labeled 0.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A Graph dataclass instance containing the points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a circular dataset where points are classified based on their distance
    from the center of the unit square. Points outside a certain radius are labeled 1,
    others labeled 0.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A Graph dataclass instance containing the points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a spiral dataset where points are classified based on their position
    in a spiral pattern. Points in one arm of the spiral are labeled 0, others labeled 1.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A Graph dataclass instance containing the points and their labels.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
