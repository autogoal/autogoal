import argparse
import sys
from autogoal.experimental.deepmatcher import DATASETS
from autogoal.experimental.deepmatcher.dataset import DeepMatcherDataset

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

