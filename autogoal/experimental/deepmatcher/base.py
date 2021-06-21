from os import mkdir, remove
from autogoal.kb import AlgorithmBase
from autogoal.kb import algorithm

from autogoal.grammar import CategoricalValue, ContinuousValue, DiscreteValue
from autogoal.kb import Sentence, Seq, Text
from autogoal.kb import Supervised, VectorCategorical, MatrixContinuousDense
from autogoal.utils import nice_repr

import deepmatcher
from pathlib import Path
import csv
import random
import string

CACHE = Path(__file__).parent / ".cache"


def buildMatchingSet(file_name):
    return deepmatcher.data.process(
        path=CACHE,
        train=file_name,
        ignore_columns=("left_id", "right_id"),
        left_prefix="left_",
        right_prefix="right_",
        label_attr="label",
        id_attr="id",
        embeddings="glove.twitter.27B.25d",
        embeddings_cache_path=(CACHE.parent / ".vector_cache"),
    )


def split_X(X):
    p = int(80 / 100 * len(X))
    return X[:p], X[p:]


def random_string():
    return "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(10)
    )


@nice_repr
class SupervisedTextMatcher(AlgorithmBase):
    """
    
    Params:
        -preprocessor: Add headers to datasets
        -attr_summarizer: The attribute summarizer. Takes in two word embedding sequences
            and summarizes the information in them to produce two summary vectors as output.
        -classifier: The neural network to perform match / mismatch classification based
            on attribute similarity representations.
        -epoch: Number of training epochs, i.e., number of times to cycle through the entire training set.
        -label_smoothing: The `label_smoothing` parameter to constructor of :class:`~deepmatcher.optim.SoftNLLLoss` criterion.
    """

    def __init__(
        self,
        preprocessor: algorithm(Seq[Seq[Text]], Seq[Seq[Text]]),
        attr_summarizer: CategoricalValue("sif", "rnn", "attention", "hybrid"),
        classifier: CategoricalValue(
            "2-layer-highway",
            "2-layer-highway-tanh",
            "3-layer-residual",
            "3-layer-residual-relu",
        ),
        epoch: DiscreteValue(min=5, max=10),
        label_smoothing: ContinuousValue(min=0.01, max=0.2),
    ):
        self.preprocessor = preprocessor
        self.attr_summarizer = attr_summarizer
        self.classifier = classifier
        self.epoch = epoch
        self.label_smoothing = label_smoothing
        self.model = None
        self._mode = "train"

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def run(
        self, X: Seq[Seq[Text]], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        if self._mode == "train":
            return self._train(X, y)
        elif self._mode == "eval":
            return self._eval(X)

    def _train(self, X, y):
        X = [[col for col in row] for row in X]

        X, vX = split_X(X)
        self.preprocessor.run(X)
        self.preprocessor.run(vX)
        assert vX

        if not Path.exists(CACHE):
            mkdir(CACHE)

        train = CACHE / f"{random_string()}.train"
        with open(train, "w") as f:
            w = csv.writer(f)
            w.writerows(X)

        val = CACHE / f"{random_string()}.val"
        with open(val, "w") as f:
            vw = csv.writer(f)
            vw.writerows(vX)

        self.model = deepmatcher.MatchingModel(
            attr_summarizer=self.attr_summarizer, classifier=self.classifier
        )
        train_dataset = buildMatchingSet(train)
        validation_dataset = buildMatchingSet(val)
        self.model.run_train(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            best_save_path=CACHE / f"best_model{random_string()}.pth",
            epochs=self.epoch,
            label_smoothing=self.label_smoothing,
        )

        remove(train)
        remove(val)
        return y

    def _eval(self, X):
        X = [[col for col in row] for row in X]
        self.preprocessor.run(X)

        eval = CACHE / f"{random_string()}.eval"
        with open(eval, "w") as f:
            w = csv.writer(f)
            w.writerows(X)

        eval_dataset = buildMatchingSet(eval)
        remove(eval)
        return [
            1 if x >= 0.5 else 0
            for x in self.model.run_prediction(eval_dataset)
            .to_dict()["match_score"]
            .values()
        ]


if __name__ == "__main__":
    test_name = "Fodors-Zagats"
    from autogoal.experimental.deepmatcher.dataset import DeepMatcherDataset
    from autogoal.experimental.deepmatcher import DATASETS

    headers, X_train, y_train, X_test, y_test = DeepMatcherDataset(
        test_name, DATASETS[test_name]
    ).load()

    @nice_repr
    class ProcessData(AlgorithmBase):
        HEADERS = headers

        def run(self, X: Seq[Seq[Text]]) -> Seq[Seq[Text]]:
            X.insert(0, ProcessData.HEADERS)
            return X

    s = SupervisedTextMatcher(
        preprocessor=ProcessData(),
        attr_summarizer="attention",
        classifier="2-layer-highway",
        epoch=2,
        label_smoothing=0.02,
    )
    s.train()
    s.run(X_train, y_train)
    s.eval()
    print(s.run(X_test, y_test)[:10])
