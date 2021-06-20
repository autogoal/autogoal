import argparse
import sys
from autogoal.experimental.deepmatcher import DATASETS
from autogoal.utils import Min, Gb
from autogoal.kb import Sentence, Seq, Supervised ,VectorCategorical
from autogoal.experimental.deepmatcher.base import SupervisedTextMatcher
from autogoal.contrib import find_classes
from autogoal.experimental.deepmatcher.dataset import DeepMatcherDataset
from autogoal.search import RichLogger

DATASET_NAMES = list(DATASETS.keys())

parser = argparse.ArgumentParser(description='DeepMatcher example')
parser.add_argument('-d', '--dataset', default=DATASET_NAMES[0], help='which dataset to use?')
parser.add_argument('-l', '--list_datasets', action='store_true', help='list all availables datasets')
args = parser.parse_args()

if args.list_datasets:
    print('Available datasets')
    print('\n'.join(DATASET_NAMES))
    sys.exit(0)

if args.dataset not in DATASET_NAMES:
    print('The given dataset is not in the options')
    sys.exit(0)

dataset = DeepMatcherDataset(args.dataset, DATASETS[args.dataset])
train, validation, test = dataset.load()

automl = AutoML(
    input = (Seq[Sentence], Supervised[VectorCategorical]),
    output = VectorCategorical,
    registry = [SupervisedTextMatcher] + find_classes(),
    evaluation_timeout = 3 * Min,
    memory_limit = 3 * Gb,
    search_timeout = 10 * Min
)

automl.fit(X_train,y_train,logger=RichLogger())
score = automl.score(X_test, y_test)
print(score)
