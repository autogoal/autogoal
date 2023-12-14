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
from autogoal.datasets import intentsemo
from autogoal.search import (
    JsonLogger,
    RichLogger,
    NSPESearch,
    PESearch
)
from autogoal_sklearn._generated import Perceptron, KNNImputer
from autogoal_sklearn._manual import ClassifierTransformerTagger, AggregatedTransformer, ClassifierTagger

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
parser.add_argument("--token", default="6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M")
parser.add_argument("--channel", default="570734906")

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
        "name": "gpu",
        "memory": 30*Gb,
        "global_timeout": 1*Hour,
        "timeout": 10*Min,
        "amount_runs": 3,
    }
]

def run_text_sentiment_emotion(configuration):
    for index in range(configuration["amount_runs"]):
        classifier = AutoML(
            search_algorithm=PESearch,
            input=(Seq[Sentence], Seq[VectorContinuous], Supervised[AggregatedVectorCategorical]),
            output=AggregatedVectorCategorical,
            registry=find_classes(),
            search_iterations=args.iterations,
            objectives=macro_f1_plain,
            cross_validation_steps=3,
            pop_size=args.popsize,
            search_timeout=configuration["global_timeout"],
            evaluation_timeout=configuration["timeout"],
            memory_limit=configuration["memory"],
        )

        X, y, _, _ = intentsemo.load(mode=intentsemo.TaskType.SentenceClassification)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_train_0, X_train_1 = zip(*X_train)
        X_train_0, X_train_1 = list(X_train_0), list(X_train_1)
        
        X_test_0, X_test_1 = zip(*X_test)
        X_test_0, X_test_1 = list(X_test_0), list(X_test_1)

        log_id = f"intentsemo-text-sentiment-emotion-{configuration['name']}-{index}"
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

        classifier.fit((X_train_0, X_train_1), y_train, logger=loggers)
        scores = classifier.score((X_test_0, X_test_1), y_test)

        # save test scores
        json_logger.append_scores(scores, classifier.best_pipelines_)

def run_sentiment_emotion(configuration):
    for index in range(configuration["amount_runs"]):
        classifier = AutoML(
            search_algorithm=PESearch,
            input=(Seq[VectorContinuous], Supervised[VectorCategorical]),
            output=VectorCategorical,
            registry=find_classes(),
            search_iterations=args.iterations,
            objectives=macro_f1_plain,
            cross_validation_steps=3,
            pop_size=args.popsize,
            search_timeout=configuration["global_timeout"],
            evaluation_timeout=configuration["timeout"],
            memory_limit=configuration["memory"],
        )

        X, y, _, _ = intentsemo.load(mode=intentsemo.TaskType.SentenceClassification)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_train_0, X_train_1 = zip(*X_train)
        X_train_0, X_train_1 = list(X_train_0), list(X_train_1)
        
        X_test_0, X_test_1 = zip(*X_test)
        X_test_0, X_test_1 = list(X_test_0), list(X_test_1)

        log_id = f"intentsemo-sentiment-emotion-{configuration['name']}-{index}"
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

        classifier.fit(X_train_1, y_train, logger=loggers)
        scores = classifier.score(X_test_1, y_test)

        # save test scores
        json_logger.append_scores(scores, classifier.best_pipelines_)

def run_text(configuration):
    for index in range(configuration["amount_runs"]):
        classifier = AutoML(
            search_algorithm=PESearch,
            input=(Seq[Sentence], Seq[VectorContinuous], Supervised[VectorCategorical]),
            output=VectorCategorical,
            registry=find_classes(exclude="Aggregated"),
            search_iterations=args.iterations,
            objectives=macro_f1_plain,
            cross_validation_steps=3,
            pop_size=args.popsize,
            search_timeout=configuration["global_timeout"],
            evaluation_timeout=configuration["timeout"],
            memory_limit=configuration["memory"],
        )

        X, y, _, _ = intentsemo.load(mode=intentsemo.TaskType.SentenceClassification)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_train_0, X_train_1 = zip(*X_train)
        X_train_0, X_train_1 = list(X_train_0), list(X_train_1)
        
        X_test_0, X_test_1 = zip(*X_test)
        X_test_0, X_test_1 = list(X_test_0), list(X_test_1)

        log_id = f"intentsemo-text-{configuration['name']}-{index}"
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

        classifier.fit(X_train_0, y_train, logger=loggers)
        scores = classifier.score(X_test_0, y_test)

        # save test scores
        json_logger.append_scores(scores, classifier.best_pipelines_)

def run_experiment(experiment, configuration, task):
    try:
        experiment(configuration)
    except Exception as e:
        print(f"Failed experiment {task}-{configuration}. Reason: {e}")
        
    print(f"Finished experiment {task}-{configuration['name']}")


def test_text_sent_emo():
    classifier = AutoML(
        search_algorithm=PESearch,
        input=(
                Seq[Sentence],
                Seq[Tensor[1, Continuous, None]],
                Supervised[AggregatedVectorCategorical],
            ),
        output=AggregatedVectorCategorical,
        registry=find_classes(exclude="TOC"),
        search_iterations=args.iterations,
        objectives=macro_f1_plain,
        cross_validation_steps=5,
        pop_size=args.popsize,
    )
    
    log_id = f"intentsemo"
    json_logger = JsonLogger(f"{log_id}.json")
    loggers = [json_logger, RichLogger()]

    from autogoal_sklearn import TfidfVectorizer, NearestCentroid, CountVectorizerTokenizeStem, HashingVectorizer, SGDClassifier
    from autogoal_nltk import MWETokenizer, ISRIStemmer, StopwordRemover
    from autogoal_contrib import AggregatedMatrixClassifier, SparseMatrixConcatenator

    pipelines = [
        Pipeline(
            algorithms=[
                HashingVectorizer(
                    alternate_sign=False,
                    binary=False,
                    lowercase=False,
                    n_features=2097151,
                    norm="l1",
                ),
                SparseMatrixConcatenator(),
                AggregatedMatrixClassifier(
                    classifier=SGDClassifier(
                    average=True,
                    early_stopping=False,
                    epsilon=0.993,
                    eta0=-0.992,
                    fit_intercept=True,
                    l1_ratio=0.6773892466070602,
                    learning_rate="optimal",
                    loss="perceptron",
                    n_iter_no_change=8,
                    penalty="l2",
                    power_t=-4.995,
                    shuffle=False,
                    tol=-0.005,
                    validation_fraction=0.006,
                ))
            ],
            input_types=(
                Seq[Sentence],
                Seq[Tensor[1, Continuous, None]],
                Supervised[AggregatedVectorCategorical],
            ),
        ),
        ]
    
    X_train, y_train, X_test, y_test = intentsemo.load(mode=intentsemo.TaskType.SentenceClassification)
    
    X_train_0, X_train_1 = zip(*X_train)
    X_train_0, X_train_1 = list(X_train_0), list(X_train_1)
    
    X_test_0, X_test_1 = zip(*X_test)
    X_test_0, X_test_1 = list(X_test_0), list(X_test_1)
    
    classifier.best_pipelines_ = pipelines
    classifier.fit_pipeline((X_train_0, X_train_1), y_train)
    scores = classifier.score((X_test_0, X_test_1), y_test)
    print("text + sent + emo:", scores)

    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)

def test_text_emo():
    classifier = AutoML(
        search_algorithm=PESearch,
        input=(
                Seq[Sentence],
                Seq[Tensor[1, Continuous, None]],
                Supervised[AggregatedVectorCategorical],
            ),
        output=AggregatedVectorCategorical,
        registry=find_classes(exclude="TOC"),
        search_iterations=args.iterations,
        objectives=macro_f1_plain,
        cross_validation_steps=5,
        pop_size=args.popsize,
    )
    
    log_id = f"intentsemo"
    json_logger = JsonLogger(f"{log_id}.json")
    loggers = [json_logger, RichLogger()]

    from autogoal_sklearn import TfidfVectorizer, NearestCentroid, CountVectorizerTokenizeStem, HashingVectorizer, SGDClassifier
    from autogoal_nltk import MWETokenizer, ISRIStemmer, StopwordRemover
    from autogoal_contrib import AggregatedMatrixClassifier, SparseMatrixConcatenator

    pipelines = [
        Pipeline(
            algorithms=[
                HashingVectorizer(
                    alternate_sign=False,
                    binary=False,
                    lowercase=False,
                    n_features=2097151,
                    norm="l1",
                ),
                SparseMatrixConcatenator(),
                AggregatedMatrixClassifier(
                    classifier=SGDClassifier(
                    average=True,
                    early_stopping=False,
                    epsilon=0.993,
                    eta0=-0.992,
                    fit_intercept=True,
                    l1_ratio=0.6773892466070602,
                    learning_rate="optimal",
                    loss="perceptron",
                    n_iter_no_change=8,
                    penalty="l2",
                    power_t=-4.995,
                    shuffle=False,
                    tol=-0.005,
                    validation_fraction=0.006,
                ))
            ],
            input_types=(
                Seq[Sentence],
                Seq[Tensor[1, Continuous, None]],
                Supervised[AggregatedVectorCategorical],
            ),
        ),
        ]
    
    X_train, y_train, X_test, y_test = intentsemo.load(mode=intentsemo.TaskType.SentenceClassification, include_sentiment=False, include_emotions=True)
    
    X_train_0, X_train_1 = zip(*X_train)
    X_train_0, X_train_1 = list(X_train_0), list(X_train_1)
    
    X_test_0, X_test_1 = zip(*X_test)
    X_test_0, X_test_1 = list(X_test_0), list(X_test_1)
    
    classifier.best_pipelines_ = pipelines
    classifier.fit_pipeline((X_train_0, X_train_1), y_train)
    scores = classifier.score((X_test_0, X_test_1), y_test)
    print("text + emo:", scores)

    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)

def test_text_sent():
    classifier = AutoML(
        search_algorithm=PESearch,
        input=(
                Seq[Sentence],
                Seq[Tensor[1, Continuous, None]],
                Supervised[AggregatedVectorCategorical],
            ),
        output=AggregatedVectorCategorical,
        registry=find_classes(exclude="TOC"),
        search_iterations=args.iterations,
        objectives=macro_f1_plain,
        cross_validation_steps=5,
        pop_size=args.popsize,
    )
    
    log_id = f"intentsemo"
    json_logger = JsonLogger(f"{log_id}.json")
    loggers = [json_logger, RichLogger()]

    from autogoal_sklearn import TfidfVectorizer, NearestCentroid, CountVectorizerTokenizeStem, HashingVectorizer, SGDClassifier
    from autogoal_nltk import MWETokenizer, ISRIStemmer, StopwordRemover
    from autogoal_contrib import AggregatedMatrixClassifier, SparseMatrixConcatenator

    pipelines = [
        Pipeline(
            algorithms=[
                HashingVectorizer(
                    alternate_sign=False,
                    binary=False,
                    lowercase=False,
                    n_features=2097151,
                    norm="l1",
                ),
                SparseMatrixConcatenator(),
                AggregatedMatrixClassifier(
                    classifier=SGDClassifier(
                    average=True,
                    early_stopping=False,
                    epsilon=0.993,
                    eta0=-0.992,
                    fit_intercept=True,
                    l1_ratio=0.6773892466070602,
                    learning_rate="optimal",
                    loss="perceptron",
                    n_iter_no_change=8,
                    penalty="l2",
                    power_t=-4.995,
                    shuffle=False,
                    tol=-0.005,
                    validation_fraction=0.006,
                ))
            ],
            input_types=(
                Seq[Sentence],
                Seq[Tensor[1, Continuous, None]],
                Supervised[AggregatedVectorCategorical],
            ),
        ),
        ]
    
    X_train, y_train, X_test, y_test = intentsemo.load(mode=intentsemo.TaskType.SentenceClassification, include_sentiment=True, include_emotions=False)
    
    X_train_0, X_train_1 = zip(*X_train)
    X_train_0, X_train_1 = list(X_train_0), list(X_train_1)
    
    X_test_0, X_test_1 = zip(*X_test)
    X_test_0, X_test_1 = list(X_test_0), list(X_test_1)
    
    classifier.best_pipelines_ = pipelines
    classifier.fit_pipeline((X_train_0, X_train_1), y_train)
    scores = classifier.score((X_test_0, X_test_1), y_test)
    print("text + sent:", scores)

    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)

def test_text():
    classifier = AutoML(
        search_algorithm=PESearch,
        input=(
                Seq[Sentence],
                Supervised[VectorCategorical],
            ),
        output=VectorCategorical,
        registry=find_classes(exclude="TOC"),
        search_iterations=args.iterations,
        objectives=macro_f1_plain,
        cross_validation_steps=5,
        pop_size=args.popsize,
    )
    
    log_id = f"intentsemo"
    json_logger = JsonLogger(f"{log_id}.json")
    loggers = [json_logger, RichLogger()]

    from autogoal_sklearn import TfidfVectorizer, NearestCentroid, CountVectorizerTokenizeStem, HashingVectorizer, SGDClassifier
    from autogoal_nltk import MWETokenizer, ISRIStemmer, StopwordRemover
    from autogoal_contrib import AggregatedMatrixClassifier, SparseMatrixConcatenator

    pipelines = [
        Pipeline(
            algorithms=[
                HashingVectorizer(
                    alternate_sign=False,
                    binary=False,
                    lowercase=False,
                    n_features=2097151,
                    norm="l1",
                ),
                SGDClassifier(
                    average=True,
                    early_stopping=False,
                    epsilon=0.993,
                    eta0=-0.992,
                    fit_intercept=True,
                    l1_ratio=0.6773892466070602,
                    learning_rate="optimal",
                    loss="perceptron",
                    n_iter_no_change=8,
                    penalty="l2",
                    power_t=-4.995,
                    shuffle=False,
                    tol=-0.005,
                    validation_fraction=0.006,
                )
            ],
            input_types=(
                Seq[Sentence],
                Supervised[VectorCategorical],
            ),
        )
        ]
    
    X_train, y_train, X_test, y_test = intentsemo.load(mode=intentsemo.TaskType.SentenceClassification)
    
    X_train_0, X_train_1 = zip(*X_train)
    X_train_0, X_train_1 = list(X_train_0), list(X_train_1)
    
    X_test_0, X_test_1 = zip(*X_test)
    X_test_0, X_test_1 = list(X_test_0), list(X_test_1)
    
    classifier.best_pipelines_ = pipelines
    classifier.fit_pipeline(X_train_0, y_train)
    scores = classifier.score(X_test_0, y_test)
    print("only_text :", scores)

    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)

def test_sent_emo():
    classifier = AutoML(
        search_algorithm=PESearch,
        input=(
                Seq[Tensor[1, Continuous, None]],
                Supervised[VectorCategorical],
            ),
        output=VectorCategorical,
        registry=find_classes(exclude="TOC"),
        search_iterations=args.iterations,
        objectives=macro_f1_plain,
        cross_validation_steps=5,
        pop_size=args.popsize,
    )
    
    log_id = f"intentsemo"
    json_logger = JsonLogger(f"{log_id}.json")
    loggers = [json_logger, RichLogger()]

    from autogoal_sklearn import TfidfVectorizer, NearestCentroid, CountVectorizerTokenizeStem, HashingVectorizer, SGDClassifier
    from autogoal_nltk import MWETokenizer, ISRIStemmer, StopwordRemover
    from autogoal_contrib import AggregatedMatrixClassifier, SparseMatrixConcatenator, MatrixBuilder

    pipelines = [
        Pipeline(
            algorithms=[
                MatrixBuilder(),
                SGDClassifier(
                    average=True,
                    early_stopping=False,
                    epsilon=0.993,
                    eta0=-0.992,
                    fit_intercept=True,
                    l1_ratio=0.6773892466070602,
                    learning_rate="optimal",
                    loss="perceptron",
                    n_iter_no_change=8,
                    penalty="l2",
                    power_t=-4.995,
                    shuffle=False,
                    tol=-0.005,
                    validation_fraction=0.006,
                )
            ],
            input_types=(
                Seq[Tensor[1, Continuous, None]],
                Supervised[VectorCategorical],
            ),
        ),
        ]
    
    X_train, y_train, X_test, y_test = intentsemo.load(mode=intentsemo.TaskType.SentenceClassification)
    
    X_train_0, X_train_1 = zip(*X_train)
    X_train_0, X_train_1 = list(X_train_0), list(X_train_1)
    
    X_test_0, X_test_1 = zip(*X_test)
    X_test_0, X_test_1 = list(X_test_0), list(X_test_1)
    
    classifier.best_pipelines_ = pipelines
    classifier.fit_pipeline(X_train_1, y_train)
    scores = classifier.score(X_test_1, y_test)
    print("sent + emo: ", scores)

    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)



tasks = {
    # "text": run_text,
    # "text-emotion-sentiment": run_text_sentiment_emotion,
    "emotion-sentiment": run_sentiment_emotion,
}

# condition = lambda x: x["name"] == args.configuration
# configuration = next(x for x in configurations if condition(x))

# initialize_cuda_multiprocessing()

# test_text_sent_emo()
# test_text_emo()
# test_text_sent()
# test_sent_emo()
test_text()

# for task in tasks.keys():
#     experiment = tasks[task]
#     run_experiment(experiment, configurations[0], task)

