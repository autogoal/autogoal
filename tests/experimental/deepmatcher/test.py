from autogoal.kb import Sentence, Seq, Supervised ,VectorCategorical
from autogoal.ml import AutoML
from autogoal.contrib import find_classes
from autogoal.utils import Min, Gb
from autogoal.kb import Seq, Word, Sentence, MatrixContinuousDense, build_pipeline_graph
from autogoal.contrib import find_classes
from autogoal.experimental.deepmatcher.base import SupervisedTextMatcher
from autogoal.search import RichLogger

from autogoal.experimental.deepmatcher.dataset import DeepMatcherDataset
from autogoal.experimental.deepmatcher import DATASETS

def test_supervised_text_matcher():
    test_name = 'Fodors-Zagats'
    X_train, y_train , X_test , y_test = DeepMatcherDataset(test_name, DATASETS[test_name]).load() # fix load

    automl = AutoML(
        input = (Seq[Sentence], Supervised[VectorCategorical]), # fix annotations
        output = VectorCategorical, # fix annotations
        registry = [SupervisedTextMatcher] + find_classes(),
        evaluation_timeout = 3 * Min,
        memory_limit = 3 * Gb,
        search_timeout = 10 * Min
    )

    automl.fit(X_train,y_train,logger=RichLogger())
    score = automl.score(X_test, y_test)
    print(score)

def test_supervised_text_matcher_pipeline():
    pipelines = build_pipeline_graph(
        input_types = (Seq[Sentence], Seq[Word]),  # fix annotations
        output_type = MatrixContinuousDense,  # fix annotations
        registry = find_classes() + [SupervisedTextMatcher],
    )
    nodes = pipelines.nodes()
    assert SupervisedTextMatcher in nodes

if __name__ == '__main__':
    test_supervised_text_matcher()
