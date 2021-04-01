"""
AutoGOAL is a Python framework for the automatic optimization, generation and learning of software pipelines.

A software pipeline is defined, for the purpose of AutoGOAL, as any software component, whether a class hierarchy,
a set of functions, or any combination thereof, that work together to solve a specific problem.
With AutoGOAL you can define a pipeline in many different ways, such that certain parts of it are configurable or
tunable, and then use search algorithms to find the best way to tune or configure it for a given problem.
"""

from autogoal import contrib
from autogoal import datasets
from autogoal import grammar
from autogoal import kb
from autogoal import ml
from autogoal import sampling
from autogoal import search
from autogoal import utils
from autogoal import logging

from autogoal.utils._helpers import optimize


# Setup logging for warning level by defautl
logging.setup(level="WARNING")
