import pickle
from autogoal.contrib.sklearn import LinearSVC, DecisionTreeClassifier
from autogoal.contrib.nltk import (
    WhitespaceTokenizer,
    PunktSentenceTokenizer,
    WordPunctTokenizer,
    TreebankWordTokenizer,
    BlanklineTokenizer,
    Doc2Vec,
)
from autogoal.contrib.wrappers import (
    MatrixBuilder,
    TensorBuilder,
    VectorColumnAggregator,
    VectorRowAggregator,
)
from autogoal.utils import Min, Hour, Gb, Sec
from autogoal.kb import Sentence, Seq, VectorCategorical, Supervised
from autogoal.ml import AutoML
from autogoal.search import (
    ConsoleLogger,
    MemoryLogger,
    ProgressLogger,
    PESearch,
    RandomSearch,
)
from autogoal.experimental.hyperopt_search import HyperoptSearch
from autogoal.contrib.gensim import Word2VecEmbedding, Word2VecSmallEmbedding

from autogoal.datasets import newsgroup20, dbpedia, imdb
from autogoal.contrib.telegram import TelegramLogger
from autogoal.experimental.keras_presets import (
    BiLSTMClassifier,
    LSTMClassifier,
    StackedBiLSTMClassifier,
    StackedLSTMClassifier,
)
from autogoal.contrib.transformers import BertEmbedding, BertTokenizeEmbedding

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("search", type=str, help="name of the search algorithm")
parser.add_argument("registry", type=str, help="name of the algorithm registry")
parser.add_argument("dataset", type=str, help="name of the dataset")
parser.add_argument("--telegramfile", help="log to telegram")
parser.add_argument("--id", help="experiment id")
parser.add_argument("--maxtime", help="maximum time in minutes id")
args = parser.parse_args()

max_time = 30 * Min
try:
    max_time = int(args.maxtime) * Min
except:
    pass

# Setting up data

## Registries
single_registry = [
    WhitespaceTokenizer,
    Word2VecSmallEmbedding,
    MatrixBuilder,
    TensorBuilder,
    LSTMClassifier,
    # LinearSVC,
    # VectorColumnAggregator,
]

small_registry = [
    # Tokenizers
    PunktSentenceTokenizer,
    WordPunctTokenizer,
    TreebankWordTokenizer,
    WhitespaceTokenizer,
    # Embeddings
    Word2VecEmbedding,
    BertEmbedding,
    # Models
    LSTMClassifier,
    BiLSTMClassifier,
    LinearSVC,
    # Wrappers
    MatrixBuilder,
    TensorBuilder,
    VectorColumnAggregator,
]

large_registry = [
    # Tokenizers
    PunktSentenceTokenizer,
    BlanklineTokenizer,
    WordPunctTokenizer,
    TreebankWordTokenizer,
    WhitespaceTokenizer,
    # Embeddings
    Word2VecEmbedding,
    BertEmbedding,
    BertTokenizeEmbedding,
    Doc2Vec,
    # Models
    LSTMClassifier,
    BiLSTMClassifier,
    StackedLSTMClassifier,
    StackedBiLSTMClassifier,
    LinearSVC,
    DecisionTreeClassifier,
    # Wrappers
    MatrixBuilder,
    TensorBuilder,
    VectorColumnAggregator,
    VectorRowAggregator,
]


registries = {"single": single_registry}

registry = registries[args.registry]

## Searches

searches = {"random": RandomSearch, "pe": PESearch, "hyperopt": HyperoptSearch}

searches_args = {
    "random": {"pop_size": 1, "search_iterations": 100},
    "pe": {"pop_size": 10, "search_iterations": 10},
    "hyperopt": {"search_registry": registry, "search_iterations": 100},
}

search = searches[args.search]
search_args = searches_args[args.search]

## Datasets
datasets = {"newsgroup20": newsgroup20, "dbpedia": dbpedia, "imdb": imdb}

dataset = datasets[args.dataset]

## Getting name identifier


run_name = (
    f"{args.search}_{args.registry}_{args.dataset}_{args.id}"
    if args.id is not None
    else f"{args.search}_{args.registry}_{args.dataset}"
)

# Reading data

X_train, y_train, X_test, y_test = dataset.load()

classifier = AutoML(
    output=VectorCategorical,
    search_algorithm=search,
    evaluation_timeout=30 * Min,
    search_timeout=None,
    memory_limit=14 * Gb,
    registry=registry,
    cross_validation_steps=2,
    early_stop=None,
    input=(Seq[Sentence], Supervised[VectorCategorical]),
    **search_args,
)

# Initializing loggers

memory = MemoryLogger()
loggers = [ProgressLogger(), ConsoleLogger(), memory]
if args.telegramfile is not None:
    with open(args.telegramfile, "rb") as file:
        telegram_data = pickle.load(file)
        telegram_logger = TelegramLogger(
            telegram_data["bot_key"], telegram_data["chat_id"], run_name,
        )
        loggers.append(telegram_logger)


# Fitting the  classifier and evaluating the final score
classifier.fit(X_train, y_train, logger=loggers)
score = classifier.score(X_test, y_test)

# Printing results and saving logger to pickle
print("FINAL RESULTS:\n\n TEST SCORE:")
print(score)
print("\nSOLUTION")
print(repr(classifier.best_pipeline_))
print("\nBEST EVALUATION EPOCH:")
print(memory.generation_best_fn)
print("\nMEAN EVALUATION EPOCH:")
print(memory.generation_mean_fn)

data = {
    "score": score,
    "solution": repr(classifier.best_pipeline_),
    "best_fns": memory.generation_best_fn,
    "mean_fn": memory.generation_mean_fn,
}

with open(f"{run_name}_result.p", "wb") as file:
    pickle.dump(data, file)
