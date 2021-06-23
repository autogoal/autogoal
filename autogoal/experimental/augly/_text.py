from autogoal.kb._semantics import Sentence, Seq
from autogoal.grammar import BooleanValue, CategoricalValue, DiscreteValue, ContinuousValue

from autogoal.utils import nice_repr

from augly.text import (
    insert_punctuation_chars,
    insert_zero_width_chars,
    replace_bidirectional,
    replace_fun_fonts,
    replace_similar_chars,
    replace_upside_down,
    simulate_typos,
    split_words,
)

from ._utils import AugLyTransformer

@nice_repr
class InsertPunctuation(AugLyTransformer):
    def __init__(
        self,
        granularity: CategoricalValue('all', 'word'),
        cadence: ContinuousValue(0.1, 1.0),
        vary_chars: BooleanValue()
        ):
        
        self.granulatity = granularity
        self.cadence = cadence
        self.vary_chars = vary_chars
        super().__init__()

    def transform(self, X, y=None):
        return insert_punctuation_chars(
            X, 
            self.granulatity,
            self.cadence,
            self.vary_chars)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Seq[Sentence]) -> Seq[Sentence]:
        return super().run(X)

@nice_repr
class InsertZeroWidth(AugLyTransformer):
    def __init__(
        self,
        granularity: CategoricalValue('all', 'word'),
        cadence: ContinuousValue(0.1, 1.0),
        vary_chars: BooleanValue()
        ):
        
        self.granulatity = granularity
        self.cadence = cadence
        self.vary_chars = vary_chars
        super().__init__()

    def transform(self, X, y=None):
        return insert_zero_width_chars(
            X, 
            self.granulatity,
            self.cadence,
            self.vary_chars)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Seq[Sentence]) -> Seq[Sentence]:
        return super().run(X)

@nice_repr
class ReplaceBidirectional(AugLyTransformer):
    def __init__(
        self,
        granularity: CategoricalValue('all', 'word'),
        split_word: BooleanValue()
        ):
        
        self.granulatity = granularity
        self.split_word = split_word
        super().__init__()

    def transform(self, X, y=None):
        return replace_bidirectional(
            X, 
            self.granulatity,
            self.split_word)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Seq[Sentence]) -> Seq[Sentence]:
        return super().run(X)

@nice_repr
class ReplaceFunFonts(AugLyTransformer):
    def __init__(
        self,
        aug_p: ContinuousValue(0, 0.6),
        aug_min: DiscreteValue(1, 5),
        aug_max: DiscreteValue(500, 10000),
        granularity: CategoricalValue('all', 'word', 'all'),
        vary_fonts: BooleanValue(),
        n: DiscreteValue(1, 3)
        ):
        
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.granularity = granularity
        self.vary_fonts = vary_fonts
        self.n = n
        super().__init__()

    def transform(self, X, y=None):
        return replace_fun_fonts(
            X, 
            self.aug_p,
            self.aug_min,
            self.aug_max,
            self.granularity,
            self.vary_fonts,
            self.n)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Seq[Sentence]) -> Seq[Sentence]:
        return super().run(X)

@nice_repr
class ReplaceSimilarChars(AugLyTransformer):
    def __init__(
        self,
        aug_char_p: ContinuousValue(0, 0.6),
        aug_word_p: ContinuousValue(0, 0.6),
        min_char: DiscreteValue(1, 4),
        aug_char_min: DiscreteValue(1, 5),
        aug_char_max: DiscreteValue(500, 10000),
        aug_word_min: DiscreteValue(1, 5),
        aug_word_max: DiscreteValue(500, 10000),
        n: DiscreteValue(1, 3)
        ):
        
        self.aug_char_p = aug_char_p
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        self.aug_char_min = aug_char_min
        self.aug_char_max = aug_char_max
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n
        super().__init__()

    def transform(self, X, y=None):
        return replace_similar_chars(
            X, 
            self.aug_char_p,
            self.aug_word_p,
            self.min_char,
            self.aug_char_min,
            self.aug_char_max,
            self.aug_word_min,
            self.aug_word_max,
            self.n)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Seq[Sentence]) -> Seq[Sentence]:
        return super().run(X)

@nice_repr
class ReplaceUpsideDown(AugLyTransformer):
    def __init__(
        self,
        aug_p: ContinuousValue(0, 0.6),
        aug_min: DiscreteValue(1, 5),
        aug_max: DiscreteValue(500, 10000),
        granularity: CategoricalValue('all', 'word', 'char'),
        n: DiscreteValue(1, 3)
        ):
        
        self.aug_p = aug_p,
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.granularity = granularity
        self.n = n
        super().__init__()

    def transform(self, X, y=None):
        return replace_upside_down(
            X, 
            self.aug_p,
            self.aug_min,
            self.aug_max,
            self.granularity,
            self.n)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Seq[Sentence]) -> Seq[Sentence]:
        return super().run(X)

@nice_repr
class SimulateTypos(AugLyTransformer):
    def __init__(
        self,
        aug_char_p: ContinuousValue(0, 0.6),
        aug_word_p: ContinuousValue(0, 0.6),
        min_char: DiscreteValue(1, 4),
        aug_char_min: DiscreteValue(1, 1),
        aug_char_max: DiscreteValue(1, 1),
        aug_word_min: DiscreteValue(1, 1),
        aug_word_max: DiscreteValue(500, 1000),
        n: DiscreteValue(1, 3)
        ):
        
        self.aug_char_p = aug_char_p
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        self.aug_char_min = aug_char_min
        self.aug_char_max = aug_char_max
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n
        super().__init__()

    def transform(self, X, y=None):
        return simulate_typos(
            X, 
            self.aug_char_p,
            self.aug_word_p,
            self.min_char,
            self.aug_char_min,
            self.aug_char_max,
            self.aug_word_min,
            self.aug_word_max,
            self.n)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Seq[Sentence]) -> Seq[Sentence]:
        return super().run(X)

@nice_repr
class SplitWords(AugLyTransformer):
    def __init__(
        self,
        aug_word_p: ContinuousValue(0, 0.6),
        min_char: DiscreteValue(1, 4),
        aug_word_min: DiscreteValue(1, 5),
        aug_word_max: DiscreteValue(500, 1000),
        n: DiscreteValue(1, 3)
        ):
        
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n
        super().__init__()

    def transform(self, X, y=None):
        return split_words(
            X, 
            self.aug_word_p,
            self.min_char,
            self.aug_word_min,
            self.aug_word_max,
            self.n)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def run(self, X: Seq[Sentence]) -> Seq[Sentence]:
        return super().run(X)