from autogoal.contrib import find_classes
from autogoal.kb import *
from autogoal.kb import build_pipelines

pipeline_space = build_pipelines(
    input=List(Sentence()),
    output=List(Flags()),
    registry=find_classes(),
)
