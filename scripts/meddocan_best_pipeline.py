# BEST MODEL OBTAINED FOR MEDDOCAN
# ________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         (None, None, 768)         0
# _________________________________________________________________
# time_distributed_495 (TimeDi (None, None, 1007)        774383
# _________________________________________________________________
# time_distributed_496 (TimeDi (None, None, 858)         864864
# _________________________________________________________________
# time_distributed_497 (TimeDi (None, None, 26)          22334
# =================================================================
# Total params: 1,661,581
# Trainable params: 1,661,581
# Non-trainable params: 0
# _________________________________________________________________

from autogoal.grammar import GraphGrammar, Path
from autogoal.contrib.keras._generated import TimeDistributed, Dense

grammar = GraphGrammar(start=Path("L1", "L2"))
grammar.add("L1", TimeDistributed, kwargs=dict(layer=Dense(units=1007)))
grammar.add("L2", TimeDistributed, kwargs=dict(layer=Dense(units=858)))

from autogoal.contrib.keras._base import KerasSequenceTagger

tagger = KerasSequenceTagger(grammar=grammar)
tagger.sample()

from autogoal.kb import (
    build_composite_list,
    build_composite_tuple,
    List,
    Word,
    Postag,
    MatrixContinuousDense,
    Tuple,
)

TupleAlgorithm = build_composite_tuple(
    0,
    Tuple(List(List(Word())), List(List(Postag()))),
    Tuple(List(MatrixContinuousDense()), List(List(Postag()))),
)

ListAlgorithm = build_composite_list(List(List(Word())), List(MatrixContinuousDense()))

from autogoal.contrib.torch._bert import BertEmbedding
from autogoal.kb._algorithm import Pipeline

pipeline = Pipeline(
    steps=[TupleAlgorithm(ListAlgorithm(BertEmbedding(verbose=False))), tagger]
)

print(pipeline)

from autogoal.datasets import meddocan

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--examples", default=None, type=int)
args = parser.parse_args()

Xtrain, Xtest, ytrain, ytest = meddocan.load(max_examples=args.examples)

pipeline.run((Xtrain, ytrain))
pipeline.send("eval")

ypred = pipeline.run((Xtest, None))

print("F1", meddocan.F1_beta(ytest, ypred))
print("Precision", meddocan.precision(ytest, ypred))
print("Recall", meddocan.recall(ytest, ypred))
