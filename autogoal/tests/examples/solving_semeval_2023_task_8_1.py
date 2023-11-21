# # Solving the MEDDOCAN challenge

# This script runs an instance of [`AutoML`](/api/autogoal.ml#automl)
# in the [MEDDOCAN 2019 challenge](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN).
# The full source code can be found [here](https://github.com/autogoal/autogoal/blob/main/docs/examples/solving_meddocan_2019.py).

# | Dataset | URL |
# |--|--|
# | MEDDOCAN 2019 | <https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN> |

# ## Experimentation parameters
#
# This experiment was run with the following parameters:
#
# | Parameter | Value |
# |--|--|
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
from autogoal.datasets import meddocan
from autogoal_transformers import BertTokenizeEmbedding, BertEmbedding
from autogoal_keras import KerasSequenceClassifier, KerasClassifier
from autogoal.datasets.semeval_2023_task_8_1 import macro_f1, macro_f1_plain, load, TaskTypeSemeval, TargetClassesMapping, SemevalDatasetSelection
from autogoal.search import (
    JsonLogger,
    RichLogger,
    PESearch,
)
from autogoal.kb import *
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import concurrent.futures
import multiprocessing
import numpy as np

# ## Parsing arguments

# Next, we parse the command line arguments to configure the experiment.

# The default values are the ones used for the experimentation reported in the paper.

import argparse

from autogoal.utils import Gb, Min, Hour

parser = argparse.ArgumentParser()
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--configuration", type=int, default=0)
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--timeout", type=int, default=30*Min)
parser.add_argument("--memory", type=int, default=20)
parser.add_argument("--popsize", type=int, default=50)
parser.add_argument("--selection", type=int, default=10)
parser.add_argument("--global-timeout", type=int, default=None)
parser.add_argument("--examples", type=int, default=None)
parser.add_argument("--token", default=None)
parser.add_argument("--channel", default=None)

args = parser.parse_args()

print(args)

# ## Experimentation

# Instantiate the classifier.
# Note that the input and output types here are defined to match the problem statement,
# i.e., entity recognition.

from autogoal_contrib import find_classes

# try token classification first

configurations = [
    {
        "name": "low-resources",
        "memory": 6*Gb,
        "global_timeout": 24*Hour,
        "timeout": 10*Min
    },
    {
        "name": "mid-resources",
        "memory": 16*Gb,
        "global_timeout": 24*Hour,
        "timeout": 20*Min
    },
    {
        "name": "high-resources",
        "memory": 32*Gb,
        "global_timeout": 24*Hour,
        "timeout": 30*Min
    }
]

tasks = [
    # "token-classification",
    "sentence-classification",
    "extended-sentence-classification",
]

def stratified_train_test_split(X, y, test_size=0.3):
    # Flatten y and remember the lengths of the sublists
    y_flat = [item for sublist in y for item in sublist]
    lengths = [len(sublist) for sublist in y]

    # Perform stratified sampling
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_index, test_index = next(sss.split(X, y_flat))

    # Split X and y
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train_flat, y_test_flat = np.array(y_flat)[train_index], np.array(y_flat)[test_index]

    # Unflatten y_train and y_test
    y_train = []
    y_test = []
    i = 0
    for length in lengths:
        if i in train_index:
            y_train.append(y_train_flat[i:i+length])
        else:
            y_test.append(y_test_flat[i:i+length])
        i += length

    return list(X_train), list(X_test), y_train, y_test

def run_token_classification(configuration):
    classifier = AutoML(
        search_algorithm=PESearch,
        input=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        output=Seq[Seq[Label]],
        registry=[BertEmbedding, KerasClassifier] + find_classes(exclude="TEC"),
        search_iterations=args.iterations,
        objectives=macro_f1,
        cross_validation_steps=1,
        pop_size=args.popsize,
        search_timeout=configuration["global_timeout"],
        evaluation_timeout=configuration["timeout"],
        memory_limit=configuration["memory"] * Gb,
    )
    
    X, y, _, _ = load(mode=TaskTypeSemeval.TokenClassification, data_option=SemevalDatasetSelection.Original)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    json_logger = JsonLogger("token-classification")
    loggers = [JsonLogger("token-classification"), RichLogger()]
    
    if args.token:
        from autogoal_telegram import TelegramLogger

        telegram = TelegramLogger(
            token=args.token,
            name=f"token-classification",
            channel=args.channel,
        )
        loggers.append(telegram)

    classifier.fit(X_train, y_train, logger=loggers)
    scores = classifier.score(X_test, y_test)
    
    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)
    
def run_sentence_classification(configuration):
    classifier = AutoML(
        search_algorithm=PESearch,
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        registry=[BertTokenizeEmbedding, KerasSequenceClassifier] + find_classes(exclude="TOC"),
        search_iterations=args.iterations,
        objectives=macro_f1_plain,
        cross_validation_steps=1,
        pop_size=args.popsize,
        search_timeout=configuration["global_timeout"],
        evaluation_timeout=configuration["timeout"],
        memory_limit=configuration["memory"] * Gb,
    )
    
    X, y, _, _ = load(mode=TaskTypeSemeval.SentenceClassification, data_option=SemevalDatasetSelection.Original, classes_mapping=TargetClassesMapping.Original)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    json_logger = JsonLogger("sentence-classification")
    loggers = [JsonLogger("sentence-classification"), RichLogger()]
    
    if args.token:
        from autogoal_telegram import TelegramLogger

        telegram = TelegramLogger(
            token=args.token,
            name=f"sentence-classification",
            channel=args.channel,
        )
        loggers.append(telegram)
        
    classifier.fit(X_train, y_train, logger=loggers)
    scores = classifier.score(X_test, y_test)
    
    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)
    
def run_extended_sentence_classification(configuration):
    classifier = AutoML(
        search_algorithm=PESearch,
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        registry=[BertTokenizeEmbedding, KerasSequenceClassifier] + find_classes(exclude="TOC"),
        search_iterations=args.iterations,
        objectives=macro_f1_plain,
        cross_validation_steps=1,
        pop_size=args.popsize,
        search_timeout=configuration["global_timeout"],
        evaluation_timeout=configuration["timeout"],
        memory_limit=configuration["memory"] * Gb,
    )
    
    X, y, _, _ = load(mode=TaskTypeSemeval.SentenceClassification, data_option=SemevalDatasetSelection.Original, classes_mapping=TargetClassesMapping.Extended)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    json_logger = JsonLogger("extended-sentence-classification")
    loggers = [JsonLogger("extended-sentence-classification"), RichLogger()]
    
    if args.token:
        from autogoal_telegram import TelegramLogger

        telegram = TelegramLogger(
            token=args.token,
            name=f"extended-sentence-classification",
            channel=args.channel,
        )
        loggers.append(telegram)
        
    classifier.fit(X_train, y_train, logger=loggers)
    scores = classifier.score(X_test, y_test)
    
    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)
    
    
# for configuration in configurations:
#     for task in tasks:
#         if task == "token-classification":
#             run_token_classification(configuration)
#         elif task == "sentence-classification":
#             run_sentence_classification(configuration)
#         elif task == "extended-sentence-classification":
#             run_extended_sentence_classification(configuration)
#         else:
#             raise Exception(f"Unknown task {task}")


def run_experiment(configuration, task):
    if task == "token-classification":
        run_token_classification(configuration)
    elif task == "sentence-classification":
        run_sentence_classification(configuration)
    elif task == "extended-sentence-classification":
        run_extended_sentence_classification(configuration)
    else:
        raise Exception(f"Unknown task {task}")
    
# Separate configurations into different lists
low_resource_configurations = [config for config in configurations if config['name'] == 'low-resources']
mid_resource_configurations = [config for config in configurations if config['name'] == 'mid-resources']
high_resource_configurations = [config for config in configurations if config['name'] == 'high-resources']
    
# Create a ProcessPoolExecutor
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Run low-resource configurations
    for configuration in low_resource_configurations:
        max_workers = 4
        for i in range(0, len(tasks), max_workers):
            executor.map(run_experiment, [configuration]*max_workers, tasks[i:i+max_workers])
            
    for future in concurrent.futures.as_completed(concurrent.futures.futures):
        pass
    print("Finished low-resource experiments (Sentence Classification)")

    # Run mid-resource configurations
    for configuration in mid_resource_configurations:
        max_workers = 2
        for i in range(0, len(tasks), max_workers):
            executor.map(run_experiment, [configuration]*max_workers, tasks[i:i+max_workers])

    for future in concurrent.futures.as_completed(concurrent.futures.futures):
        pass
    print("Finished mid-resource experiments (Sentence Classification)")

    # Run high-resource configurations
    for configuration in high_resource_configurations:
        max_workers = 1
        for i in range(0, len(tasks), max_workers):
            executor.map(run_experiment, [configuration]*max_workers, tasks[i:i+max_workers])
    
    for future in concurrent.futures.as_completed(concurrent.futures.futures):
        pass
    print("Finished high-resource experiments (Sentence Classification)")
            
# run_experiment(configurations[0], tasks[1])