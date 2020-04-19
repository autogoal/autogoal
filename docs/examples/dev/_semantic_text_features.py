from autogoal.contrib import find_classes
from autogoal.kb import *
from autogoal.kb import build_pipelines, build_pipeline_graph

from autogoal.contrib.spacy import SpacyNLP
from autogoal.contrib._wrappers import FlagsMerger

import logging

logging.basicConfig(level=logging.INFO)


pipeline_space = build_pipeline_graph(
    input=List(Sentence()),
    output=MatrixContinuousDense(),
    registry=find_classes(),
    # registry=[SpacyNLP, FlagsMerger],
    # max_list_depth=1,
)


for i in range(10):
    pipeline = pipeline_space.sample()
    print(pipeline)
