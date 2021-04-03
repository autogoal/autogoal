"""
AutoGOAL is a Python framework for the automatic optimization, generation and learning of software pipelines.

A software pipeline is defined, for the purpose of AutoGOAL, as any software component, whether a class hierarchy,
a set of functions, or any combination thereof, that work together to solve a specific problem.
With AutoGOAL you can define a pipeline in many different ways, such that certain parts of it are configurable or
tunable, and then use search algorithms to find the best way to tune or configure it for a given problem.
"""

# This is the top level module for AutoGOAL, it's what you we get when we `import autogoal`.
# By default, Python won't import submodules, hence, if we want `autogoal.*` to work we'll have
# to import all submodules here manually.

# These four submodules include the low-level components of AutoGOAL
# from which we can build all the core functionality.

# The [`grammar`](ref:autogoal.grammar.__init__)
# and [`sampling`](ref:autogoal.sampling.__init__) submodules allows us to
# define search spaces with arbitrary structure,
# and automatically create instances of different types of objects by sampling from them.
from autogoal import grammar
from autogoal import sampling

# The [`kb`](ref:autgoal.kb.__init__) submodule allows us to
# define algorithms based on input and output types that
# and combine them automatically intro pipeline graphs.
from autogoal import kb

# The [`search`](ref:autogoal.search.__init__) submodule contains search
# strategies to optimize in different hyper-parameter spaces.
from autogoal import search

# With these low level structures, we can build the core functionality of AutoGOAL, the AutoML algorithm.
# The [`ml`](ref:autogoal.ml.__init__) submodule contains the definition of the [`AutoML`](ref:autogoal.ml._automl:AutoML) class
# and related functionality.
from autogoal import ml

from autogoal import utils
from autogoal import logging

from autogoal import datasets

from autogoal import contrib

from autogoal.utils._helpers import optimize


# Setup logging for warning level by default

logging.setup(level="WARNING")
