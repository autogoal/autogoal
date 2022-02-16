from autogoal.kb import *
from autogoal.grammar import CategoricalValue, BooleanValue
from autogoal.utils import nice_repr
from autogoal.kb import AlgorithmBase

import numpy as np


@nice_repr
class VectorRowAggregator(AlgorithmBase):
    def __init__(self, mode: CategoricalValue("mean", "max")):
        self.mode = mode

    def run(self, input: Seq[VectorContinuous]) -> VectorContinuous:
        input = np.vstack(input)

        if self.mode == "mean":
            return input.mean(axis=1)
        elif self.mode == "max":
            return input.max(axis=1)

        raise ValueError("Invalid mode: %s" % self.mode)


@nice_repr
class VectorColumnAggregator(AlgorithmBase):
    def __init__(self, mode: CategoricalValue("mean", "max")):
        self.mode = mode

    def run(self, input: Seq[VectorContinuous]) -> VectorContinuous:
        input = np.vstack(input)

        if self.mode == "mean":
            return input.mean(axis=0)
        elif self.mode == "max":
            return input.max(axis=0)

        raise ValueError("Invalid mode: %s" % self.mode)


@nice_repr
class BasePadder(AlgorithmBase):
    def __init__(self, method):
        self.method = method
        self._mode = "train"
        self._length = -1

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def _run(self, input):
        if self._mode == "train":
            self._find_length(input)
            return self._pad_data(input)

        elif self._mode == "eval":
            return self._pad_data(input)

    def _find_length(self, X):
        selected = np.inf if self.method == "min" else 0
        for sample in X:
            this_len = len(sample)
            if self.method == "min":
                selected = this_len if selected > this_len else selected
            elif self.method == "max":
                selected = this_len if selected < this_len else selected
            elif self.method == "mean":
                selected += this_len
        if self.method == "mean":
            self._length = int(selected / len(X))
        else:
            self._length = selected

    def _pad_data(self, X):
        new_X = []
        for sample in X:
            if len(sample) >= self._length:
                new_X.append(sample[: self._length])
            else:
                pads_len = self._length - len(sample)
                pads = [(0, pads_len) if i == 0 else (0, 0) for i in range(sample.ndim)]
                new_X.append(np.pad(sample, pads, "constant", constant_values=(0, 0)))
        return new_X


@nice_repr
class VectorPadder(BasePadder):
    """

    """

    def __init__(self, method: CategoricalValue("min", "mean", "max")):
        super().__init__(method)

    def run(self, input: Seq[VectorContinuous]) -> UniformSeq[VectorContinuous]:
        return self._run(input)


@nice_repr
class MatrixPadder(BasePadder):
    """

    """

    def __init__(self, method: CategoricalValue("min", "mean", "max")):
        super().__init__(method)

    def run(
        self, input: Seq[MatrixContinuousDense]
    ) -> UniformSeq[MatrixContinuousDense]:
        return self._run(input)


@nice_repr
class MatrixBuilder(BasePadder):
    """
    Builds a matrix from a list of vectors.

    ##### Examples

    ```python
    >>> import numpy as np
    >>> x1 = np.asarray([1,2,3])
    >>> x2 = np.asarray([2,3,4])
    >>> x3 = np.asarray([3,4,5])
    >>> MatrixBuilder().run([x1, x2, x3])
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])

    ```
    """

    def __init__(self, method: CategoricalValue("min", "mean", "max")):
        super().__init__(method)

    def run(self, input: Seq[VectorContinuous]) -> MatrixContinuousDense:
        padded = self._run(input)
        return np.vstack(padded)


@nice_repr
class TensorBuilder(BasePadder):
    """
    Builds a 3D tensor from a list of matrices.

    ##### Examples

    ```python
    >>> import numpy as np
    >>> x1 = np.asarray([[1,2],[3,4]])
    >>> x2 = np.asarray([[2,3],[4,5]])
    >>> x3 = np.asarray([[3,4],[5,6]])
    >>> TensorBuilder().run([x1, x2, x3])
    array([[[1, 2],
            [3, 4]],
    <BLANKLINE>
           [[2, 3],
            [4, 5]],
    <BLANKLINE>
           [[3, 4],
            [5, 6]]])

    ```
    """

    def __init__(self, method: CategoricalValue("min")):
        super().__init__(method)

    def run(self, input: Seq[MatrixContinuousDense]) -> Tensor3:
        padded = self._run(input)
        return np.vstack([np.expand_dims(m, axis=0) for m in padded])


@nice_repr
class FlagsMerger(AlgorithmBase):
    def run(self, input: Seq[FeatureSet]) -> FeatureSet:
        result = {}

        for d in input:
            result.update(d)

        return result


@nice_repr
class MultipleFeatureExtractor(AlgorithmBase):
    def __init__(
        self,
        extractors: Distinct(
            algorithm(Word, FeatureSet), exceptions=["MultipleFeatureExtractor"]
        ),
        merger: algorithm(Seq[FeatureSet], FeatureSet),
    ):
        self.extractors = extractors
        self.merger = merger

    def run(self, input: Word) -> FeatureSet:
        flags = [extractor.run(input) for extractor in self.extractors]
        return self.merger.run(flags)


@nice_repr
class SentenceFeatureExtractor(AlgorithmBase):
    def __init__(
        self,
        tokenizer: algorithm(Sentence, Seq[Word]),
        feature_extractor: algorithm(Word, FeatureSet),
        include_text: BooleanValue(),
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.include_text = include_text

    def run(self, input: Sentence) -> FeatureSet:
        tokens = self.tokenizer.run(input)
        flags = [self.feature_extractor(w) for w in tokens]

        if self.include_text:
            return {
                f"{w}|{f}": v for w, flag in zip(tokens, flags) for f, v in flag.items()
            }
        else:
            return {f: v for flag in flags for f, v in flag.items()}


@nice_repr
class DocumentFeatureExtractor(AlgorithmBase):
    def __init__(
        self,
        tokenizer: algorithm(Document, Seq[Sentence]),
        feature_extractor: algorithm(Sentence, FeatureSet),
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def run(self, input: Document) -> Seq[FeatureSet]:
        tokens = self.tokenizer.run(input)
        flags = [self.feature_extractor(w) for w in tokens]
        return


# @nice_repr
# class TextEntityEncoder(AlgorithmBase):
#     """
#     Convierte una oración en texto plano más la lista de entidades
#     reconocidas en la oración, en una lista de tokens con sus respectivas
#     categorias BILOUV.
#     """

#     def __init__(self, tokenizer: algorithm(Sentence, Seq[Word])):
#         self.tokenizer = tokenizer

#     def run(
#         self, input: Tuple(Sentence, Seq[Entity])
#     ) -> Tuple(Seq[Word], Seq[Label]:
#         pass


# @nice_repr
# class TextRelationEncoder(AlgorithmBase):
#     """
#     Convierte una oración en texto plano y una lista de relaciones
#     que se cumplen entre entidades, en una lista de ejemplos
#     por cada oración.
#     """

#     def __init__(self,
#         tokenizer: algorithm(Sentence, Seq[Word]),
#         token_feature_extractor: algorithm(Word, FeatureSet),
#         # token_sentence_encoder: algorithm(Word, )
#     ):
#         pass

#     def run(
#         self, input: Tuple(Sentence, Seq[Tupl](Entity(), Entity(), Category())))
#     ) -> Tuple(Seq[Vect]r()), CategoricalVector()):
#         pass
