import nltk
from gensim.models.doc2vec import Doc2Vec as _Doc2Vec

from numpy import inf, nan

from autogoal.contrib.nltk._builder import NltkTokenizer, NltkTagger
from autogoal.contrib.sklearn._builder import SklearnTransformer, SklearnWrapper
from autogoal.grammar import Boolean, Categorical, Continuous, Discrete
from autogoal.kb import *
from autogoal.kb._data import *
from autogoal.utils import nice_repr

nltk.download("wordnet")
nltk.download("sentiwordnet")
nltk.download("stopwords")

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords, wordnet


@nice_repr
class Doc2Vec(_Doc2Vec, SklearnTransformer):
    def __init__(
        self,
        dm: Discrete(min=0, max=2),
        dbow_words: Discrete(min=-100, max=100),
        dm_concat: Discrete(min=-100, max=100),
        dm_tag_count: Discrete(min=0, max=2),
        alpha: Continuous(min=0.001, max=0.075),
        epochs: Discrete(min=2, max=10),
        window: Discrete(min=2, max=10),
        inner_tokenizer: algorithm(Sentence(), List(Word())),
        inner_stemmer: algorithm(Word(), Stem()),
        inner_stopwords: algorithm(List(Word()), List(Word())),
        lowercase: Boolean(),
        stopwords_remove:Boolean(),
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
        # Data must be turned to tagged data as TaggedDocument(List(Token), Tag)
        # Tag use to be an unique integer

        from gensim.models.doc2vec import TaggedDocument as _TaggedDocument

        tagged_data = [_TaggedDocument(self.tokenize(X[i]), str(i)) for i in range(len(X))]

        self.build_vocab(tagged_data)
        return self.train(
            tagged_data, total_examples=self.corpus_count, epochs=self.epochs
        )

    def transform(self, X, y=None):
        return [self.infer_vector(x) for x in X]

    def run(self, input: List(Sentence())) -> MatrixContinuousDense():
        """This methods receive a document list and transform this into a dense continuous matrix.
       """
        return SklearnTransformer.run(self, input)


@nice_repr
class StopwordRemover:
    def __init__(
        self,
        language: Categorical(
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
        self.words = stopwords.words(language)
        SklearnWrapper.__init__(self)

    def _train(self, input):
        return [word for word in input if word not in self.words]

    def _eval(self, input):
        return [word for word in input if word not in self.words]

    def run(self, input: List(Word())) -> List(Word()):
        """This methods receive a word list list and transform this into a word list list without stopwords.
       """
        return SklearnTransformer.run(self, input)

    def __str__(self):
        name = StopwordRemover.__name__
        return f"{name}({self.language})"


@nice_repr
class TextLowerer:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y=None)
        return self.transform(X)

    def transform(self, X, y=None):
        # Considering data as list of raw documents
        return [str.lower(x) for x in X]


@nice_repr
class WordnetConcept:
    """Find a word in Wordnet and return a List of Synset de Wordnet
    """

    def __init__(self):
        pass

    def run(self, input: Word(domain="general", language="english")) -> Synset():
        """Find a word in Wordnet and return a List of Synset de Wordnet
        """
        synsets = wordnet.synsets(input)
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
class SentimentWord:
    """Find a word in SentiWordnet and return a List of sentiment of the word.
    """

    def __init__(self):
        pass

    def run(self, input: Synset(domain="general", language="english")) -> Sentiment():
        """Find a word in SentiWordnet and return a List of sentiment of the word.
        """
        swn_synset = swn.senti_synset(input)

        sentiment = {}
        sentiments["positive"] = i.pos_score()
        sentiments["negative"] = i.neg_score()

        return sentiment


from nltk.chunk.named_entity import NEChunkParserTagger as _NEChunkParserTagger

@nice_repr
class NEChunkParserTagger(NltkTagger):
    def __init__(self,):
        self.tagger = _NEChunkParserTagger
        self.values = dict()

        NltkTagger.__init__(self)

    def run(self, input: List(List(Postag()))) -> List(List(Chunktag())):
        return NltkTagger.run(self, input)


@nice_repr
class GlobalChunker(SklearnWrapper):
    def __init__(
        self,
        inner_trained_pos_tagger: algorithm(List(Word()), List(Postag())),
        inner_chunker: algorithm(List(List(Postag())), List(List(Chunktag())))
    ):
        self.inner_trained_pos_tagger = inner_trained_pos_tagger
        self.inner_chunker = inner_chunker

        SklearnWrapper.__init__(self)

    def _train(self, input):
        X, y = input

        postagged_sentences = []
        for i in range(len(X)):
            sentence = []
            for itoken in range(len(X[i])):
                x_sent = X[i]
                word = x_sent[itoken]
                sentence.append(word)

            postag_sentence = self.inner_trained_pos_tagger.run((sentence, sentence))[0]
            tagged_sentence = [ (sentence[k], postag_sentence[k][1]) for k in range(len(sentence))]
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
            tagged_sentence = [ ((sentence[k], postag_sentence[k][1]), tags[k]) for k in range(len(sentence))]
            if tagged_sentence:
                tagged_sentences.append(tagged_sentence)

        return self.inner_chunker.run((postagged_sentences, tagged_sentences))

    def _eval(self, input):
        X, y = input

        postagged_document = []
        for i in range(len(X)):
            sentence = []
            for itoken in range(len(X[i])):
                x_sent = X[i]
                word = x_sent[itoken]
                sentence.append(word)

            postag_sentence = self.inner_trained_pos_tagger.run((sentence, sentence))[0]
            tagged_sentence = [ (sentence[k], postag_sentence[k][1]) for k in range(len(sentence))]
            if tagged_sentence:
                postagged_document.append(tagged_sentence)

        return self.inner_chunker.run((postagged_document, y))

    def run(self, input: List(List(Word()))) -> List(List(Chunktag())):
        return SklearnWrapper.run(self, input)
