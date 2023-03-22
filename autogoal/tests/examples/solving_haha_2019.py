# # Solving the HAHA challenge

# This script runs an instance of [`AutoML`](/api/autogoal.ml#automl)
# in the [HAHA 2019 challenge](https://www.fing.edu.uy/inco/grupos/pln/haha/index.html#data).
# The full source code can be found [here](https://github.com/autogoal/autogoal/blob/main/docs/examples/solving_haha_2019.py).

# The dataset used is:

# | Dataset | URL |
# |--|--|
# | HAHA 2019 | <https://www.fing.edu.uy/inco/grupos/pln/haha/index.html#data> |

# ## Experimentation parameters
#
# This experiment was run with the following parameters:
#
# | Parameter | Value |
# |--|--|
# | Total epochs         | 1      |
# | Maximum iterations   | 10000  |
# | Timeout per pipeline | 30 min |
# | Global timeout       | -      |
# | Max RAM per pipeline | 20 GB  |
# | Population size      | 50     |
# | Selection (k-best)   | 10     |
# | Early stop           |-       |

# The experiments were run in the following hardware configurations
# (allocated indistinctively according to available resources):

# | Config | CPU | Cache | Memory | HDD |
# |--|--|--|--|--|
# | **A** | 12 core Intel Xeon Gold 6126 | 19712 KB |  191927.2MB | 999.7GB  |
# | **B** | 6 core Intel Xeon E5-1650 v3 | 15360 KB |  32045.5MB  | 2500.5GB |
# | **C** | Quad core Intel Core i7-2600 |  8192 KB |  15917.1MB  | 1480.3GB |

# !!! note
#     The hardware configuration details were extracted with `inxi -CmD` and summarized.

# ## Relevant imports

# Most of this example follows the same logic as the [UCI example](/examples/solving_uci_datasets).
# First the necessary imports

from autogoal.ml import AutoML
from autogoal.datasets import haha
from autogoal.search import (
    PESearch,
    RichLogger,
)
from autogoal.kb import Seq, Sentence, VectorCategorical, Supervised
from autogoal.contrib import find_classes
from sklearn.metrics import f1_score

# Next, we parse the command line arguments to configure the experiment.

# ## Parsing arguments

# The default values are the ones used for the experimentation reported in the paper.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--timeout", type=int, default=60)
parser.add_argument("--memory", type=int, default=2)
parser.add_argument("--popsize", type=int, default=50)
parser.add_argument("--selection", type=int, default=10)
parser.add_argument("--global-timeout", type=int, default=None)
parser.add_argument("--examples", type=int, default=None)
parser.add_argument("--token", default=None)
parser.add_argument("--channel", default=None)

args = parser.parse_args()

print(args)

# The next line will print all the algorithms that AutoGOAL found
# in the `contrib` library, i.e., anything that could be potentially used
# to solve an AutoML problem.

for cls in find_classes():
    print("Using: %s" % cls.__name__)

# ## Experimentation

# Instantiate the classifier.
# Note that the input and output types here are defined to match the problem statement,
# i.e., text classification.

classifier = AutoML(
    search_algorithm=PESearch,
    input=(Seq[Sentence], Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_iterations=args.iterations,
    score_metric=f1_score,
    errors="warn",
    pop_size=args.popsize,
    search_timeout=args.global_timeout,
    evaluation_timeout=args.timeout,
    memory_limit=args.memory * 1024**3,
)

loggers = [RichLogger()]

if args.token:
    from autogoal.contrib.telegram import TelegramLogger

    telegram = TelegramLogger(
        token=args.token,
        name=f"HAHA",
        channel=args.channel,
    )
    loggers.append(telegram)

# Finally, loading the HAHA dataset, running the `AutoML` instance,
# and printing the results.

X_train, y_train, X_test, y_test = haha.load(max_examples=args.examples)

classifier.fit(X_train, y_train, logger=loggers)
score = classifier.score(X_test, y_test)

print(score)
