import nltk
from gensim.models.doc2vec import Doc2Vec as _Doc2Vec

from numpy import inf, nan

from autogoal.contrib.nltk._builder import NltkTokenizer, NltkTagger
from autogoal.contrib.sklearn._builder import SklearnTransformer, SklearnWrapper
from autogoal.grammar import (
    BooleanValue,
    CategoricalValue,
    ContinuousValue,
    DiscreteValue,
)
from autogoal.kb import *
from autogoal.utils import nice_repr
from autogoal.kb import AlgorithmBase


@nice_repr
class Doc2Vec(_Doc2Vec, SklearnTransformer):
    def __init__(
        self,
        dm: DiscreteValue(min=0, max=2),
        dbow_words: DiscreteValue(min=-100, max=100),
        dm_concat: DiscreteValue(min=-100, max=100),
        dm_tag_count: DiscreteValue(min=0, max=2),
        alpha: ContinuousValue(min=0.001, max=0.075),
        epochs: DiscreteValue(min=2, max=10),
        window: DiscreteValue(min=2, max=10),
        inner_tokenizer: algorithm(Sentence, Seq[Word]),
        inner_stemmer: algorithm(Word, Stem),
        inner_stopwords: algorithm(Seq[Word], Seq[Word]),
        lowercase: BooleanValue(),
        stopwords_remove: BooleanValue(),
    ):

        self.inner_tokenizer = inner_tokenizer
        self.inner_stemmer = inner_stemmer
        self.inner_stopwords = inner_stopwords
        self.lowercase = lowercase
        self.stopwords_remove = stopwords_remove

        super().__init__(
            dm=dm,
            dbow_words=dbow_words,
            dm_concat=dm_concat,
            dm_tag_count=dm_tag_count,
            alpha=alpha,
            epochs=epochs,
            window=window,
        )

    def tokenize(self, sentence):
        sentence = sentence.lower() if self.lowercase else sentence
        tokens = self.inner_tokenizer.run(sentence)
        tokens = self.inner_stopwords.run(sentence) if self.stopwords_remove else tokens
        return [self.inner_stemmer.run(token) for token in tokens]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y):
        # Data must be turned to tagged data as TaggedDocument(Seq[Token), Tag)
        # Tag use to be an unique integer

        from gensim.models.doc2vec import TaggedDocument as _TaggedDocument

        tagged_data = [
            _TaggedDocument(self.tokenize(X[i]), str(i)) for i in range(len(X))
        ]

        self.build_vocab(tagged_data)
        return self.train(
            tagged_data, total_examples=self.corpus_count, epochs=self.epochs
        )

    def transform(self, X, y=None):
        return [self.infer_vector(x) for x in X]

    def run(self, input: Seq[Sentence]) -> MatrixContinuousDense:
        """This methods receive a document list and transform this into a dense continuous matrix.
       """
        return SklearnTransformer.run(self, input)


@nice_repr
class StopwordRemover(SklearnTransformer):
    def __init__(
        self,
        language: CategoricalValue(
            "danish",
            "dutch",
            "english",
            "finnish",
            "french",
            "german",
            "hungarian",
            "italian",
            "norwegian",
            "portuguese",
            "russian",
            "spanish",
            "swedish",
            "turkish",
        ),
    ):
        self.language = language
        from nltk.corpus import stopwords

        self.words = stopwords.words(language)
        SklearnTransformer.__init__(self)

    def fit_transform(self, X, y=None):
        return [word for word in input if word not in self.words]

    def transform(self, X, y=None):
        return self.fit_transform(X, y)

    def run(self, input: Seq[Word]) -> Seq[Word]:
        """This methods receive a word list list and transform this into a word list list without stopwords.
       """
        return SklearnTransformer.run(self, input)


@nice_repr
class TextLowerer(SklearnTransformer):
    def fit_transform(self, X, y=None):
        self.fit(X, y=None)
        return self.transform(X)

    def transform(self, X, y=None):
        # Considering data as list of raw documents
        return [str.lower(x) for x in X]


@nice_repr
class WordnetConcept(AlgorithmBase):
    """Find a word in Wordnet and return a List of Synset de Wordnet
    """

    def __init__(self):
        from nltk.corpus import wordnet

        self.wordnet = wordnet

    def run(self, input: Word) -> Seq[Synset]:
        """Find a word in Wordnet and return a List of Synset de Wordnet
        """
        synsets = self.wordnet.synsets(input)
        names_synsets = []
        for i in synsets:
            names_synsets.append(i.name())

        return names_synsets


# @nice_repr
# class ConvertSynset2Word:
#     """Recive a Synset of nltk and return de Lemma of this
#     """

#     def __init__(self):
#         pass

#     def run(self, input: Synset(domain="general", language="english")) -> Word():
#         """Recive a Synset of nltk and return de Lemma of this
#         """
#         return Lemma(input)


@nice_repr
class SentimentWord(AlgorithmBase):
    """Find a word in SentiWordnet and return a List of sentiment of the word.
    """

    def __init__(self):
        from nltk.corpus import sentiwordnet

        self.swn = sentiwordnet

    def run(self, input: Synset) -> Sentiment:
        """Find a word in SentiWordnet and return a List of sentiment of the word.
        """
        swn_synset = self.swn.senti_synset(input)

        sentiment = {}
        sentiment["positive"] = swn_synset.pos_score()
        sentiment["negative"] = swn_synset.neg_score()

        return sentiment


from nltk.chunk.named_entity import NEChunkParserTagger as _NEChunkParserTagger


@nice_repr
class NEChunkParserTagger(NltkTagger):
    def __init__(self,):
        self.tagger = _NEChunkParserTagger
        self.values = dict()

        NltkTagger.__init__(self)

    def run(self, input: Seq[Postag]) -> Seq[Chunktag]:
        return NltkTagger.run(self, input)


@nice_repr
class GlobalChunker(SklearnWrapper):
    def __init__(
        self,
        inner_trained_pos_tagger: algorithm(Seq[Word], Seq[Postag]),
        inner_chunker: algorithm(Seq[Postag], Seq[Chunktag]),
    ):
        self.inner_trained_pos_tagger = inner_trained_pos_tagger
        self.inner_chunker = inner_chunker

        SklearnWrapper.__init__(self)

    def _train(self, X, y):
        postagged_sentences = []
        for i in range(len(X)):
            sentence = []
            for itoken in range(len(X[i])):
                x_sent = X[i]
                word = x_sent[itoken]
                sentence.append(word)

            postag_sentence = self.inner_trained_pos_tagger.run((sentence, sentence))[0]
            tagged_sentence = [
                (sentence[k], postag_sentence[k][1]) for k in range(len(sentence))
            ]
            if tagged_sentence:
                postagged_sentences.append(tagged_sentence)

        tagged_sentences = []
        for i in range(len(y)):
            sentence = []
            tags = []
            for itoken in range(len(y[i])):
                y_sent = y[i]
                word, tag = y_sent[itoken]
                sentence.append(word)
                tags.append(tag)

            postag_sentence = self.inner_trained_pos_tagger.run((sentence, sentence))[0]
            tagged_sentence = [
                ((sentence[k], postag_sentence[k][1]), tags[k])
                for k in range(len(sentence))
            ]
            if tagged_sentence:
                tagged_sentences.append(tagged_sentence)

        return self.inner_chunker.run((postagged_sentences, tagged_sentences))

    def _eval(self, X, y=None):
        postagged_document = []
        for i in range(len(X)):
            sentence = []
            for itoken in range(len(X[i])):
                x_sent = X[i]
                word = x_sent[itoken]
                sentence.append(word)

            postag_sentence = self.inner_trained_pos_tagger.run((sentence, sentence))[0]
            tagged_sentence = [
                (sentence[k], postag_sentence[k][1]) for k in range(len(sentence))
            ]
            if tagged_sentence:
                postagged_document.append(tagged_sentence)

        return self.inner_chunker.run((postagged_document, y))

    def run(
        self, X: Seq[Seq[Word]], y: Supervised[Seq[Seq[Chunktag]]]
    ) -> Seq[Chunktag]:
        return SklearnWrapper.run(self, input)


@nice_repr
class FeatureSeqExtractor(AlgorithmBase):
    """
    A simple feature extractor for tokenized sentences based on NLTK.

    It receives a tokenized sentence (i.e., a list of str) and outputs a 
    list of syntactic features. For each word in the sentence, it builds a
    dictionary of features of that word, plus features of surrounding words
    in a pre-defined window (of size 1 to 5).

    **Parameters**

    * `extract_word`: whether to extract words as features.
    * `window_size`: size of the window around the current word to also look for features.

    ```python
    >>> extractor = FeatureSeqExtractor(window_size=2)
    >>> extractor.run(["Hello", "World"])
    [{'word': 'Hello', 'word+1': 'World'}, {'word': 'World', 'word-1': 'Hello'}]

    ```
    """

    def __init__(
        self, extract_word: BooleanValue() = True, window_size: DiscreteValue(0, 5) = 0,
    ):
        self.extract_word = extract_word
        self.window_size = window_size

    def extract_features(self, w):
        features = {}

        if self.extract_word:
            features["word"] = w

        return features

    def run(self, sentence: Seq[Word]) -> Seq[FeatureSet]:
        features = []

        for w in sentence:
            features.append(self.extract_features(w))

        expanded_features = []

        for i, f in enumerate(features):
            expanded = dict(f)

            for j in range(i - self.window_size, i + self.window_size + 1):
                if j == i:
                    continue

                ff = features[j] if 0 <= j < len(features) else {}
                idx = f"+{j-i}" if j > i else str(j - i)

                for k, v in ff.items():
                    expanded[k + idx] = v

            expanded_features.append(expanded)

        return expanded_features


__all__ = [
    "Doc2Vec",
    "StopwordRemover",
    "TextLowerer",
    "WordnetConcept",
    "SentimentWord",
    "NEChunkParserTagger",
    "GlobalChunker",
    "FeatureSeqExtractor",
]
