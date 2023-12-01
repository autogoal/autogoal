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
from autogoal.ml.metrics import peak_ram_usage, evaluation_time
from autogoal.utils import initialize_cuda_multiprocessing
from autogoal_transformers import BertTokenizeSequenceEmbedding, BertEmbedding, BertSequenceEmbedding
from autogoal_keras import KerasSequenceClassifier
from autogoal.datasets.semeval_2023_task_8_1 import macro_f1, macro_f1_plain, load, TaskTypeSemeval, TargetClassesMapping, SemevalDatasetSelection
from autogoal.search import (
    JsonLogger,
    RichLogger,
    NSPESearch,
)
from autogoal_sklearn._generated import Perceptron, KNNImputer
from autogoal_sklearn._manual import ClassifierTransformerTagger, AggregatedTransformer

from autogoal.kb import *
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
from collections import Counter

# ## Parsing arguments

# Next, we parse the command line arguments to configure the experiment.

# The default values are the ones used for the experimentation reported in the paper.

import argparse

from autogoal.utils import Gb, Min, Hour

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="token")
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--configuration", type=str, default="cpu")
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--timeout", type=int, default=30*Min)
parser.add_argument("--memory", type=int, default=20)
parser.add_argument("--popsize", type=int, default=20)
parser.add_argument("--selection", type=int, default=5)
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
        "name": "cpu",
        "memory": 50*Gb,
        "global_timeout": 72*Hour,
        "timeout": 30*Min,
        "complexity_objective" : peak_ram_usage
    },
    {
        "name": "gpu",
        "memory": 50*Gb,
        "global_timeout": 72*Hour,
        "timeout": 90*Min,
        "complexity_objective" : evaluation_time
    }
]

tasks = [
    "token-classification",
    "sentence-classification",
    # "extended-sentence-classification",
]

def stratified_train_test_token_split(X, y, test_size=0.3):
    counts = [dict(Counter(sublist)) for sublist in y]



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

def run_token_classification(configuration, index):
    classifier = AutoML(
        search_algorithm=NSPESearch,
        input=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        output=Seq[Seq[Label]],
        registry=[AggregatedTransformer, KNNImputer, Perceptron, ClassifierTransformerTagger, BertEmbedding, BertSequenceEmbedding] + find_classes(exclude="Stopword"),
        search_iterations=args.iterations,
        objectives=(macro_f1, configuration["complexity_objective"]),
        maximize=(True, False),
        cross_validation_steps=1,
        pop_size=args.popsize,
        search_timeout=configuration["global_timeout"],
        evaluation_timeout=configuration["timeout"],
        memory_limit=configuration["memory"],
    )

    X_train, y_train, X_test, y_test = load(mode=TaskTypeSemeval.TokenClassification, data_option=SemevalDatasetSelection.Original)

    log_id = f"token-classification-{configuration['name']}"
    json_logger = JsonLogger(f"{log_id}.json")
    loggers = [json_logger, RichLogger()]

    if args.token:
        from autogoal_telegram import TelegramLogger

        telegram = TelegramLogger(
            token=args.token,
            name=log_id,
            channel=args.channel,
        )
        loggers.append(telegram)

    classifier.fit(X_train, y_train, logger=loggers)
    scores = classifier.score(X_test, y_test)

    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)

def run_sentence_classification(configuration, index):
    classifier = AutoML(
        search_algorithm=NSPESearch,
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        registry=[KerasSequenceClassifier, BertTokenizeSequenceEmbedding] + find_classes(),
        search_iterations=args.iterations,
        objectives=(macro_f1_plain, configuration["complexity_objective"]),
        maximize=(True, False),
        cross_validation_steps=1,
        pop_size=args.popsize,
        search_timeout=configuration["global_timeout"],
        evaluation_timeout=configuration["timeout"],
        memory_limit=configuration["memory"],
    )

    X, y, _, _ = load(mode=TaskTypeSemeval.SentenceClassification, data_option=SemevalDatasetSelection.Original, classes_mapping=TargetClassesMapping.Original)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    log_id = f"sentence-classification-{configuration['name']}"
    json_logger = JsonLogger(f"{log_id}.json")
    loggers = [json_logger, RichLogger()]

    if args.token:
        from autogoal_telegram import TelegramLogger

        telegram = TelegramLogger(
            token=args.token,
            name=log_id,
            channel=args.channel,
        )
        loggers.append(telegram)

    classifier.fit(X_train, y_train, logger=loggers)
    scores = classifier.score(X_test, y_test)

    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)

def run_extended_sentence_classification(configuration, index):
    classifier = AutoML(
        search_algorithm=NSPESearch,
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        registry=[BertTokenizeEmbedding, KerasSequenceClassifier] + find_classes(exclude="TOC"),
        search_iterations=args.iterations,
        objectives=(macro_f1_plain, configuration["complexity_objective"]),
        maximize=(True, False),
        cross_validation_steps=1,
        pop_size=args.popsize,
        search_timeout=configuration["global_timeout"],
        evaluation_timeout=configuration["timeout"],
        memory_limit=configuration["memory"],
    )

    X, y, _, _ = load(mode=TaskTypeSemeval.SentenceClassification, data_option=SemevalDatasetSelection.Original, classes_mapping=TargetClassesMapping.Extended)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    log_id = f"extended-sentence-classification-{configuration['name']}"
    json_logger = JsonLogger(f"{log_id}.json")
    loggers = [json_logger, RichLogger()]

    if args.token:
        from autogoal_telegram import TelegramLogger

        telegram = TelegramLogger(
            token=args.token,
            name=log_id,
            channel=args.channel,
        )
        loggers.append(telegram)

    classifier.fit(X_train, y_train, logger=loggers)
    scores = classifier.score(X_test, y_test)

    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)

def run_experiment(configuration, task, index):
    print(f"Started experiment {task}-{configuration['name']}-{index}")

    if task == "token-classification":
        try:
            run_token_classification(configuration, index)
        except Exception as e:
            print(f"Failed experiment {task}-{configuration}. Reason: {e}")

    elif task == "sentence-classification":
        try:
            run_sentence_classification(configuration, index)
        except Exception as e:
            print(f"Failed experiment {task}-{configuration}. Reason: {e}")

    elif task == "extended-sentence-classification":
        try:
            run_extended_sentence_classification(configuration, index)
        except Exception as e:
            print(f"Failed experiment {task}-{configuration}. Reason: {e}")
    else:
        raise Exception(f"Unknown task {task}")

    print(f"Finished experiment {task}-{configuration['name']}-{index}")


condition = lambda x: x["name"] == args.configuration
configuration = next(x for x in configurations if condition(x))

if args.configuration == "gpu":
    initialize_cuda_multiprocessing()
    
for exp in ["sentence", "token"]:
    run_experiment(configuration, f"{exp}-classification", args.id)
        
# if args.experiment == "token":
#     run_experiment(configuration, "token-classification", args.id)
# elif args.experiment == "sentence":
#     run_experiment(configuration, "sentence-classification", args.id)

