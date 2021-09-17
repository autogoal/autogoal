from autogoal.utils import nice_repr
from autogoal.kb._semantics import Text
from autogoal.grammar import (
    BooleanValue,
    CategoricalValue,
    DiscreteValue,
    ContinuousValue,
)

from augly.text import transforms
from augly.text.transforms import BaseTransform

from _util import AugLyTransformer



@nice_repr
class AugLyTextTransformer(AugLyTransformer):
    """
    Base class for augLy text transformers
    """

    def run(self, X: Text) -> Text:
        if self._transformer is None:
            self._transformer = self.get_transformer()
        return self._transformer(Text)


@nice_repr
class InsertPunctuationCharsTransformer(AugLyTransformer):
    '''
    Inserts punctuation characters in each input text
    '''

    def __init__(
        self,
        granularity: CategoricalValue("all", "word"),
        cadence: ContinuousValue(0.1, 1.0),
        vary_chars: BooleanValue(),
    ):
        super().__init__()
        self.granulatity = granularity
        self.cadence = cadence
        self.vary_chars = vary_chars

    def get_transformer(self):
        return transforms.InsertPunctuationChars(
            self.granulatity,
            self.cadence,
            self.vary_chars,
        )

@nice_repr
class InsertWhitespaceCharsTransformer(AugLyTransformer):
    '''
     Inserts whitespace characters in each input text
    '''

    def __init__(
        self,
        granularity: CategoricalValue("all", "word"),
        cadence: ContinuousValue(0.1, 1.0),
        vary_chars: BooleanValue(),
    ):
        super().__init__()
        self.granulatity = granularity
        self.cadence = cadence
        self.vary_chars = vary_chars

    def get_transformer(self):
        return transforms.InsertWhitespaceChars(
            self.granulatity,
            self.cadence,
            self.vary_chars,
        )


@nice_repr
class InsertZeroWidthTransformer(AugLyTransformer):
    '''
    Inserts zero-width characters in each input text
    '''

    def __init__(
        self,
        granularity: CategoricalValue("all", "word"),
        cadence: ContinuousValue(0.1, 1.0),
        vary_chars: BooleanValue(),
    ):
        super().__init__()
        self.granulatity = granularity
        self.cadence = cadence
        self.vary_chars = vary_chars

    def get_transformer(self):
        return transforms.InsertZeroWidthChars(
            self.granulatity,
            self.cadence,
            self.vary_chars,
        )


@nice_repr
class ReplaceBidirectionalTransformer(AugLyTransformer):
    '''
    Reverses each word (or part of the word) in each input text and uses
    bidirectional marks to render the text in its original order. It reverses
    each word separately which keeps the word order even when a line wraps
    '''
    def __init__(
        self,
        granularity: CategoricalValue("all", "word"),
        split_word: BooleanValue(),
    ):
        super().__init__()
        self.granulatity = granularity
        self.split_word = split_word

    def get_transformer(self, X, y=None):
        return transforms.ReplaceBidirectional(
            self.granulatity,
            self.split_word,
        )


@nice_repr
class ReplaceFunFontsTransformer(AugLyTransformer):
    '''
    Replaces words or characters depending on the granularity with fun fonts applied
    '''
    def __init__(
        self,
        aug_p: ContinuousValue(0, 1),
        aug_min: DiscreteValue(1, 10),
        aug_max: DiscreteValue(1, 10000),
        granularity: CategoricalValue("all", "word"),
        vary_fonts: BooleanValue(),
        n: DiscreteValue(1, 10),
    ):
        super().__init__()
        self.aug_p = aug_p
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.granularity = granularity
        self.vary_fonts = vary_fonts
        self.n = n

    def get_transformer(self, X, y=None):
        return transforms.ReplaceFunFonts(
            X,
            self.aug_p,
            self.aug_min,
            self.aug_max,
            self.granularity,
            self.vary_fonts,
            self.n,
        )


@nice_repr
class ReplaceSimilarCharsTransformer(AugLyTransformer):
    '''
    Replaces letters in each text with similar characters
    '''
    def __init__(
        self,
        aug_char_p: ContinuousValue(0, 0.6),
        aug_word_p: ContinuousValue(0, 0.6),
        min_char: DiscreteValue(1, 4),
        aug_char_min: DiscreteValue(1, 5),
        aug_char_max: DiscreteValue(100, 1000),
        aug_word_min: DiscreteValue(1, 5),
        aug_word_max: DiscreteValue(100, 1000),
        n: DiscreteValue(1, 3),
    ):
        super().__init__()
        self.aug_char_p = aug_char_p
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        self.aug_char_min = aug_char_min
        self.aug_char_max = aug_char_max
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n

    def get_transformer(self, X, y=None):
        return transforms.ReplaceSimilarChars(
            self.aug_char_p,
            self.aug_word_p,
            self.min_char,
            self.aug_char_min,
            self.aug_char_max,
            self.aug_word_min,
            self.aug_word_max,
            self.n,
        )


@nice_repr
class ReplaceSimilarUnicodeChars(AugLyTransformer):
    '''
    Replaces letters in each text with similar unicodes
    '''
    def __init__(
        self,
        aug_char_p: ContinuousValue(0, 0.6),
        aug_word_p: ContinuousValue(0, 0.6),
        min_char: DiscreteValue(1, 4),
        aug_char_min: DiscreteValue(1, 5),
        aug_char_max: DiscreteValue(100, 1000),
        aug_word_min: DiscreteValue(1, 5),
        aug_word_max: DiscreteValue(100, 1000),
        n: DiscreteValue(1, 3),
    ):
        super().__init__()
        self.aug_char_p = aug_char_p
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        self.aug_char_min = aug_char_min
        self.aug_char_max = aug_char_max
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n

    def get_transformer(self, X, y=None):
        return transforms.ReplaceSimilarChars(
            self.aug_char_p,
            self.aug_word_p,
            self.min_char,
            self.aug_char_min,
            self.aug_char_max,
            self.aug_word_min,
            self.aug_word_max,
            self.n,
        )


@nice_repr
class ReplaceUpsideDownTransformer(AugLyTransformer):
    def __init__(
        self,
        aug_p: ContinuousValue(0, 0.6),
        aug_min: DiscreteValue(1, 5),
        aug_max: DiscreteValue(500, 10000),
        granularity: CategoricalValue("all", "word", "char"),
        n: DiscreteValue(1, 3),
    ):
        super().__init__()
        self.aug_p = (aug_p,)
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.granularity = granularity
        self.n = n

    def get_transformer(self, X, y=None):
        return transforms.ReplaceUpsideDown(
            self.aug_p,
            self.aug_min,
            self.aug_max,
            self.granularity,
            self.n,
        )


@nice_repr
class SimulateTyposTransformer(AugLyTransformer):
    def __init__(
        self,
        aug_char_p: ContinuousValue(0, 0.6),
        aug_word_p: ContinuousValue(0, 0.6),
        min_char: DiscreteValue(1, 4),
        aug_char_min: DiscreteValue(1, 1),
        aug_char_max: DiscreteValue(1, 1),
        aug_word_min: DiscreteValue(1, 1),
        aug_word_max: DiscreteValue(500, 1000),
        n: DiscreteValue(1, 3),
    ):

        super().__init__()
        self.aug_char_p = aug_char_p
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        self.aug_char_min = aug_char_min
        self.aug_char_max = aug_char_max
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n

    def get_transformer(self):
        return transforms.SimulateTypos(
            self.aug_char_p,
            self.aug_word_p,
            self.min_char,
            self.aug_char_min,
            self.aug_char_max,
            self.aug_word_min,
            self.aug_word_max,
            self.n,
        )


@nice_repr
class SplitWordsTransformer(AugLyTransformer):
    def __init__(
        self,
        aug_word_p: ContinuousValue(0, 0.6),
        min_char: DiscreteValue(1, 4),
        aug_word_min: DiscreteValue(1, 5),
        aug_word_max: DiscreteValue(500, 1000),
        n: DiscreteValue(1, 3),
    ):
        super().__init__()
        self.aug_word_p = aug_word_p
        self.min_char = min_char
        self.aug_word_min = aug_word_min
        self.aug_word_max = aug_word_max
        self.n = n

    def get_transformer(self):
        return transforms.SplitWords(
            
            self.aug_word_p,
            self.min_char,
            self.aug_word_min,
            self.aug_word_max,
            self.n,
        )