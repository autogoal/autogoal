from autogoal.kb import Sentence, Seq, Supervised ,VectorCategorical, AlgorithmBase
from autogoal.ml import AutoML
from autogoal.contrib import find_classes
from autogoal.utils import Min, Gb
from autogoal.kb import Seq, Word, Text, Sentence, MatrixContinuousDense, build_pipeline_graph
from autogoal.contrib import find_classes
from autogoal.experimental.deepmatcher.base import SupervisedTextMatcher
from autogoal.search import RichLogger
from autogoal.utils import nice_repr

from autogoal.experimental.deepmatcher.dataset import DeepMatcherDataset
from autogoal.experimental.deepmatcher import DATASETS


def test_supervised_text_matcher():
    test_name = 'Fodors-Zagats'
    headers, X_train, y_train , X_test , y_test = DeepMatcherDataset(test_name, DATASETS[test_name]).load()

    @nice_repr
    class ProcessData(AlgorithmBase):
        HEADERS=headers
        def run(self, X: Seq[Seq[Text]]) -> Seq[Seq[Text]]:
            X.insert(0, ProcessData.HEADERS)
            return X

    automl = AutoML(
        input = (Seq[Seq[Text]], Supervised[VectorCategorical]),
        output = VectorCategorical,
        registry = [SupervisedTextMatcher,ProcessData] + find_classes(),
        evaluation_timeout = 3 * Min,
        memory_limit = 3 * Gb,
        search_timeout = 10 * Min
    )

    automl.fit(X_train,y_train,logger=RichLogger())
    score = automl.score(X_test, y_test)
    print(score)

def test_supervised_text_matcher_pipeline():
    pipelines = build_pipeline_graph(
        input_types = (Seq[Seq[Text]], Supervised[VectorCategorical]),
        output_type = VectorCategorical,
        registry = find_classes() + [SupervisedTextMatcher],
    )
    nodes = pipelines.nodes()
    assert SupervisedTextMatcher in nodes

if __name__ == '__main__':
    # test_supervised_text_matcher_pipeline() # add ProcessData in registry
    test_supervised_text_matcher()
