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

from time import sleep
from autogoal.ml import AutoML
from autogoal.datasets import meddocan
from autogoal.ml.metrics import peak_ram_usage, evaluation_time
from autogoal_transformers import BertTokenizeSequenceEmbedding, BertEmbedding
from autogoal_keras import KerasSequenceClassifier, KerasClassifier
from autogoal.datasets.semeval_2023_task_8_1 import macro_f1, weighted_f1, macro_f1_plain, weighted_f1_plain, load, TaskTypeSemeval, TargetClassesMapping, SemevalDatasetSelection
from autogoal.search import (
    JsonLogger,
    RichLogger,
    NSPESearch,
)
from autogoal_sklearn._generated import  TfidfVectorizer, MultinomialNB, MinMaxScaler, Perceptron, KNNImputer, StandardScaler, PassiveAggressiveClassifier, LinearSVC,SVC,NuSVC,DecisionTreeClassifier, LogisticRegression
from autogoal_sklearn._manual import ClassifierTransformerTagger, ClassifierTagger, AggregatedTransformer
from autogoal_nltk import WordPunctTokenizer, TweetTokenizer

from autogoal.kb import *
from autogoal.kb._algorithm import make_seq_algorithm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import concurrent.futures
import multiprocessing
import numpy as np
from collections import Counter

# ## Parsing arguments

# Next, we parse the command line arguments to configure the experiment.

# The default values are the ones used for the experimentation reported in the paper.

import argparse

from autogoal.utils import Gb, Min, Hour, initialize_cuda_multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="token")
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--configuration", type=int, default=0)
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
    # {
    #     "name": "high-resources",
    #     "memory": 50*Gb,
    #     "global_timeout": 48*Hour,
    #     "timeout": 60*Min
    # }
    # {
    #     "name": "high-resources",
    #     "memory": 50*Gb,
    #     "global_timeout": 72*Hour,
    #     "timeout": 30*Min
    # }
    {
        "name": "high-resources-gpu",
        "memory": 50*Gb,
        "global_timeout": 38*Hour,
        "timeout": 90*Min
    }
    
]

tasks = [
    "token-classification",
    "sentence-classification",
    # "extended-sentence-classification",
]

def run_token_classification(configuration, index):
    classifier = AutoML(
        search_algorithm=NSPESearch,
        input=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        output=Seq[Seq[Label]],
        registry=[AggregatedTransformer, KNNImputer, Perceptron, ClassifierTransformerTagger, BertEmbedding] + find_classes(exclude="CRFTagger|Stopword"),
        search_iterations=args.iterations,
        objectives=(macro_f1, weighted_f1),
        maximize=(True, False),
        cross_validation_steps=1,
        pop_size=args.popsize,
        search_timeout=configuration["global_timeout"],
        evaluation_timeout=configuration["timeout"],
        memory_limit=configuration["memory"],
    )

    X_train, y_train, X_test, y_test = load(mode=TaskTypeSemeval.TokenClassification, data_option=SemevalDatasetSelection.Original)

    log_id = f"token-classification-{configuration['name']}-{index}"
    json_logger = JsonLogger(f"{log_id}.json")
    loggers = [json_logger, RichLogger()]
    
    from autogoal_sklearn import TfidfVectorizer, NearestCentroid, CountVectorizerTokenizeStem, CRFTagger
    from autogoal_nltk import MWETokenizer, ISRIStemmer, StopwordRemover, FeatureSeqExtractor

    pipelines = [
        Pipeline(
            algorithms=[
                make_seq_algorithm(
                    FeatureSeqExtractor(
                        extract_word=True, 
                        window_size=5
                    )
                )(), 
                CRFTagger(
                    algorithm="ap"
                )
            ], 
            input_types=(
                Seq[Seq[Word]], 
                Supervised[Seq[Seq[Label]]]
            )
        ), 
        Pipeline(
            algorithms=[
                make_seq_algorithm(
                    FeatureSeqExtractor(
                        extract_word=False, 
                        window_size=4
                    )
                )(), 
                CRFTagger(
                    algorithm="pa"
                )
            ], 
            input_types=(
                Seq[Seq[Word]], 
                Supervised[Seq[Seq[Label]]]
            )
        ),
        Pipeline(
            algorithms=[
                make_seq_algorithm(
                    FeatureSeqExtractor(
                        extract_word=True, 
                        window_size=1
                    )
                )(), 
                CRFTagger(
                    algorithm="pa"
                )
            ], 
            input_types=(
                Seq[Seq[Word]], 
                Supervised[Seq[Seq[Label]]]
            )
        ),
        Pipeline(
            algorithms=[
                make_seq_algorithm(FeatureSeqExtractor(extract_word=False, window_size=3))(),
                CRFTagger(algorithm="arow"),
            ],
            input_types=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        ),
        Pipeline(
            algorithms=[
                make_seq_algorithm(FeatureSeqExtractor(extract_word=True, window_size=0))(),
                CRFTagger(algorithm="l2sgd"),
            ],
            input_types=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        ),
        Pipeline(
            algorithms=[
                make_seq_algorithm(FeatureSeqExtractor(extract_word=True, window_size=3))(),
                CRFTagger(algorithm="ap"),
            ],
            input_types=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        ),
        Pipeline(
            algorithms=[
                make_seq_algorithm(FeatureSeqExtractor(extract_word=True, window_size=0))(),
                CRFTagger(algorithm="pa"),
            ],
            input_types=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        ),
        Pipeline(
            algorithms=[
                make_seq_algorithm(FeatureSeqExtractor(extract_word=True, window_size=5))(),
                CRFTagger(algorithm="pa"),
            ],
            input_types=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        ),
        Pipeline(
            algorithms=[
                make_seq_algorithm(FeatureSeqExtractor(extract_word=True, window_size=2))(),
                CRFTagger(algorithm="pa"),
            ],
            input_types=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        )]
    
    classifier.best_pipelines_ = pipelines
    classifier.fit_pipeline(X_train, y_train)
    print("trained")
    scores = classifier.score(X_test, y_test)
    print(scores)

    # save test scores
    json_logger.append_scores(scores, classifier.best_pipelines_)

def run_sentence_classification(configuration, index):
    classifier = AutoML(
        search_algorithm=NSPESearch,
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        registry=find_classes(exclude="TOC"),
        search_iterations=args.iterations,
        objectives=(macro_f1_plain, weighted_f1_plain),
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

    from autogoal_sklearn import TfidfVectorizer, NearestCentroid, CountVectorizerTokenizeStem
    from autogoal_nltk import MWETokenizer, ISRIStemmer, StopwordRemover

    pipelines = [
        Pipeline(
            algorithms=[
                TfidfVectorizer(
                    binary=True,
                    lowercase=True,
                    smooth_idf=False,
                    sublinear_tf=True,
                    use_idf=False,
                ),
                LogisticRegression(
                    C=9.991,
                    dual=False,
                    fit_intercept=True,
                    multi_class="multinomial",
                    penalty="l2",
                ),
            ],
            input_types=[Seq[Sentence], Supervised[VectorCategorical]],
        ), 
        
        Pipeline(
            algorithms=[
                TfidfVectorizer(
                    binary=True,
                    lowercase=True,
                    smooth_idf=False,
                    sublinear_tf=True,
                    use_idf=True,
                ),
                NearestCentroid(),
            ],
            input_types=[Seq[Sentence], Supervised[VectorCategorical]],
        ),
        Pipeline(
            algorithms=[
                CountVectorizerTokenizeStem(
                    lowercase=False,
                    stopwords_remove=True,
                    binary=True,
                    inner_tokenizer=MWETokenizer(),
                    inner_stemmer=ISRIStemmer(),
                    inner_stopwords=StopwordRemover(language="russian"),
                ),
                NearestCentroid(),
            ],
            input_types=[Seq[Sentence], Supervised[VectorCategorical]],
        )
        ]
    
    classifier.best_pipelines_ = pipelines
    classifier.fit_pipeline(X_train, y_train)
    scores = classifier.score(X_test, y_test)
    print(scores)

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

    print(f"Finished experiment {task}-{configuration['name']}-{index}")

# Separate configurations into different lists
high_resource_configurations = [config for config in configurations if config['name'] == 'high-resources']


# Create a ProcessPoolExecutor
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     for configuration in high_resource_configurations:
#         futures = [executor.submit(run_experiment, configuration, task, 0) for task in tasks]
        
#         for future in concurrent.futures.as_completed(futures):
#             # If you need to use the result of the task
#             result = future.result()

# initialize_cuda_multiprocessing()
# run_experiment(configurations[0], "sentence-classification", args.id)
# [(0.485842966380485, 0.7352862431969472), (0.4182583506298185, 0.625048113270782), (0.4023407780366961, 0.6277978814542059)]

run_experiment(configurations[0], "token-classification", args.id)

# if args.experiment == "token":
#     run_experiment(configurations[0], "token-classification", args.id)
# elif args.experiment == "sentence":
#     run_experiment(configurations[0], "sentence-classification", args.id)

