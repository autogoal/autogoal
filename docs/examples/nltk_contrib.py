from autogoal.kb import List, Sentence, MatrixContinuousSparse
from autogoal.contrib.nltk import find_classes
from autogoal.contrib.sklearn._manual import CountVectorizerNoTokenize
from autogoal.kb import build_pipelines


for cls in find_classes():
    print(cls)


pipelines = build_pipelines(
    input=List(Sentence()),
    output=MatrixContinuousSparse(),
    registry=find_classes() + [CountVectorizerNoTokenize],
)


for i in range(10):
    print(pipelines.sample())
