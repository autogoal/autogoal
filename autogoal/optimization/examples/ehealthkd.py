# coding: utf-8

import functools
import pprint
import bisect
import os
import pprint
import random
import string
import nltk
import spacy
import unicodedata
import gensim
import yaml
import numpy as np
import warnings
import sys

from scipy import sparse as sp
from pathlib import Path

from sklearn_crfsuite.estimator import CRF
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder

from gensim.models import Word2Vec
import gensim.downloader as api

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Embedding, LSTM, Input, concatenate
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from ..ge import Grammar, PGE, Individual, InvalidPipeline
from ..datasets.ehealthkd import Dataset, Keyphrase, Relation

from ..utils import szip, sdiv


class TassGrammar(Grammar):
    def __init__(self):
        super().__init__()

        self.stemmer = SnowballStemmer("spanish")
        self.spacy_nlp = spacy.load('es')

    def grammar(self):
        return {
            'Pipeline' : 'Repr A B C | Repr AB C | Repr A BC | Repr ABC',
            'ABC'      : 'Class',
            'BC'       : 'Class',
            'AB'       : 'Class | Seq',
            'A'        : 'Class | Seq',
            'B'        : 'Class',
            'C'        : 'Class',

            # Sequence algorithms
            'Seq'      : 'HMM', # | crf',
            'HMM'      : 'HMMdec HMMalp',
            'HMMdec'   : 'viterbi | bestfirst',
            'HMMalp'   : 'f(0.01, 10)',

            # Classifiers
            'Class'    : 'LR | nb | SVM | dt | NN',

            # Classic classifiers
            'LR'       : 'Reg Penalty',
            'Reg'      : 'f(0.01,100)',
            'Penalty'  : 'l1 | l2',
            'SVM'      : 'Kernel',
            'Kernel'   : 'linear | rbf | poly',

            # Generic neural networks
            'NN'       : 'Drop CVLayers DLayers FLayer | Drop RLayers DLayers FLayer | Drop DLayers FLayer',
            'Drop'     : 'f(0.1,0.5)',
            # Convolutional layers
            'CVLayers' : 'Count MinFilter MaxFilter FormatCon',
            'Count'    : 'i(1,5)',
            # Con el objetivo de eliminar la recursividad de la gramática y controlar el tamaño de las capas
            # se define un espacio genérico donde no se define cada capa de la red, sino que se define una
            # arquitectura de alto nivel que restringe las posibles redes. No aparecen representadas todas
            # las posiblidades pero este subconjunto de ellas es semáticamente más interesante.
            'MinFilter': 'i(1,5)',
            'MaxFilter': 'i(1,5)',
            # Todos los filtros con el mismo tamaño, un filtro para cada tamaño y tamaños de filtros aleatorios
            'FormatCon': 'same | all | rand',
            # Recursive layers
            'RLayers'  : 'Size',
            'Size'     : 'i(10,100)',
            # Dense layers
            'DLayers'  : 'Count MaxSize MinSize FormatDen Act',
            'Act'      : 'sigmoid | relu | tanh',
            'MinSize'  : 'i(10,100)',
            'MaxSize'  : 'i(10,100)',
            # las capas van creciendo de tamaño del min al max, disminuyendo del max al min, todas del mismo tamaño
            'FormatDen': 'grow | shrink | same',
            # Final layer
            'FLayer'   : 'sigmoid',

            # Text representation
            'Repr'     : 'Prep Token MulWords SemFeat PosPrep Embed',
            'Prep'     : 'DelPunt StripAcc',
            'Token'    : 'wordTok',
            'DelPunt'  : 'yes | no',
            'StripAcc' : 'yes | no',
            'PosPrep'  : 'StopW Stem',
            'Stem'     : 'yes | no',
            'StopW'    : 'yes | no',
            'SemFeat'  : 'PosTag Dep UMLS SNOMED',
            'PosTag'   : 'yes | no',
            'Dep'      : 'yes | no',
            'UMLS'     : 'yes | no',
            'SNOMED'   : 'yes | no',
            'MulWords' : 'collocations | postag Ngram | none',
            'Ngram'    : 'i(2,4)',
            'Embed'    : 'wordVec | onehot | none',
        }

    def evaluate(self, ind:Individual):
        FAST = 'fast' in sys.argv
        TEST = 'test' in sys.argv

        # load training data
        dataset_path = Path.cwd() / 'hpopt' / 'datasets' / 'ehealthkd'
        dataset = Dataset()

        for file in (dataset_path / 'training').iterdir():
            if file.name.startswith('input'):
                dataset.load(file)

                if FAST and len(dataset) >= 100:
                    break

        if FAST:
            dataset.validation_size = int(0.2 * len(dataset))
        else:
            validation = dataset_path / 'develop' / 'input_develop.txt'
            dataset.validation_size = dataset.load(validation)

            if TEST:
                test = dataset_path / 'test' / 'input_scenario1.txt'
                dataset.validation_size = dataset.load(test)

        return self._pipeline(ind, dataset.clone())

    def _pipeline(self, ind, dataset):
        # 'Pipeline' : 'Repr A B C | Repr AB C |  Repr A BC | Repr ABC',
        choice = ind.choose('A B C', 'AB C', 'A BC', 'ABC')

        dataset = self._repr(ind, dataset)

        try:
            if choice == 'A B C':
                # Ejecutar tareas A, B y C en secuencia
                self._a(ind, dataset)
                self._b(ind, dataset)
                self._c(ind, dataset)
            elif choice == 'AB C':
                # Ejecutar tareas AB juntas y C en secuencia
                self._ab(ind, dataset)
                self._c(ind, dataset)
            elif choice == 'A BC':
                # Ejecutar tarea A y luego BC
                self._a(ind, dataset)
                self._bc(ind, dataset)
            else:
                # Ejecutar Tarea ABC junta
                self._abc(ind, dataset)

            return self._score(dataset)
        except ValueError as e:
            if 'must be non-negative' in str(e):
                raise InvalidPipeline(str(e))
            else:
                raise e

    def _score(self, dataset:Dataset):
        # assert len(train_labels) == len(val_labels)
        # assert len(train_relations) == len(val_relations)

        # score counts
        correctA = 0
        partialA = 0
        missingA = 0
        spuriousA = 0

        correctB = 0
        incorrectB = 0

        correctC = 0
        missingC = 0
        spuriousC = 0

        _, dev = dataset.split()

        for actual in dev.sentences:
            self.log("\n" + actual.text)
            predicted = actual.invert()

            for phrase in actual.keyphrases:
                match = predicted.find_keyphrase(id=phrase.id)

                if match:
                    correctA += 1

                    if match.label == phrase.label:
                        self.log("Correct keyphrase:", phrase, match)
                        correctB += 1
                    else:
                        self.log("Incorrect keyphrase:", phrase, match)
                        incorrectB += 1
                else:
                    self.log("Missing keyphrase:", phrase)
                    missingA += 1

            for phrase in predicted.keyphrases:
                if not actual.find_keyphrase(id=phrase.id):
                    self.log("Spurious keyphrase:", phrase)
                    spuriousA += 1

            for relation in actual.relations:
                match = predicted.find_relation(relation.origin, relation.destination, relation.label)

                if match:
                    self.log("Correct relation:", relation, match)
                    correctC += 1
                else:
                    self.log("Missing relation:", relation)
                    missingC += 1

            for relation in predicted.relations:
                match = actual.find_relation(relation.origin, relation.destination, relation.label)

                if not match:
                    self.log("Spurious relation:", relation)
                    spuriousC += 1

        precA = sdiv(correctA + 0.5 * partialA, correctA + partialA + spuriousA)
        recA = sdiv(correctA + 0.5 * partialA, correctA + partialA + missingA)

        precC = sdiv(correctC, correctC + spuriousC)
        recC = sdiv(correctC, correctC + missingC)

        self.log('correctA', correctA)
        self.log('correctB', correctB)
        self.log('correctC', correctC)
        self.log('incorrectB', incorrectB)
        self.log('missingA', missingA)
        self.log('missingC', missingC)
        self.log('partialA', partialA)
        self.log('spuriousA', spuriousA)
        self.log('spuriousC', spuriousC)

        self.log("[*] Task A: %0.2f" % sdiv(2 * precA * recA, precA + recA))
        self.log("[*] Task B: %0.2f" % sdiv(correctB, correctB + incorrectB))
        self.log("[*] Task C: %0.2f" % sdiv(2 * precC * recC, precC + recC))

        top = (correctA + 0.5 * partialA + correctB + correctC)
        spr = (correctA + partialA + correctB + incorrectB + spuriousA + correctC + spuriousC)
        precision = sdiv(top, spr)
        msn = (correctA + partialA + correctB + incorrectB + missingA + correctC + missingC)
        recall = sdiv(top, msn)

        return sdiv(2 * precision * recall, precision + recall)

    def log(self, *args, **kwargs):
        if '-v' in sys.argv:
            print(*args, **kwargs)

    def _repr(self, i, dataset:Dataset):
        # 'Prep Token SemFeat PosPrep MulWords Embed',
        dataset = self._prep(i, dataset)
        dataset = self._token(i, dataset)
        dataset = self._mulwords(i, dataset)
        dataset = self._semfeat(i, dataset)
        dataset = self._posprep(i, dataset)
        dataset = self._embed(i, dataset)

        return dataset

    def _prep(self, i, dataset:Dataset):
        #'DelPunt StripAcc'
        dataset = self._delpunt(i, dataset)
        return self._stripacc(i, dataset)

    def _delpunt(self, i, dataset:Dataset):
        #yes | no
        if i.nextbool():
            for sentence in dataset.sentences:
                sentence.text = sentence.text.translate({c:" " for c in string.punctuation})

        return dataset

    def _stripacc(self, i, dataset:Dataset):
        #yes | no
        if i.nextbool():
            for sentence in dataset.sentences:
                sentence.text = gensim.utils.deaccent(sentence.text)

        return dataset

    def _token(self, i, dataset:Dataset):
        ids = max(k.id for sentence in dataset.sentences for k in sentence.keyphrases) * 10

        for sentence in dataset.sentences:
            for token in self.spacy_nlp(sentence.text):
                features = dict(
                    norm=token.norm_,
                    pos=token.pos_,
                    tag=token.tag_,
                    dep=token.dep_,
                    vector=token.vector
                )
                start = token.idx
                end = start + len(token.text)
                match = sentence.find_keyphrase(start=start, end=end)

                if match:
                    label = match.label
                    id = match.id
                else:
                    label = ''
                    id = ids
                    ids += 1

                keyword = Keyphrase(sentence, features, label, id, start, end)
                keyword.spacy_token = [token]
                sentence.tokens.append(keyword)

        dataset.max_length = max(len(s) for s in dataset.sentences)
        return dataset

    def _posprep(self, i, dataset):
        self._stopw(i, dataset)
        self._stem(i, dataset)

        return dataset

    def _stem(self, i, dataset:Dataset):
        if i.nextbool():
            for sentence in dataset.sentences:
                for token in sentence.tokens:
                    token.features['norm'] = self.stemmer.stem(token.features['norm'])

    def _stopw(self, i, dataset:Dataset):
        if i.nextbool():
            sw = set(stopwords.words('spanish'))
        else:
            sw = set()

        for sentence in dataset.sentences:
            sentence.tokens = [t for t in sentence.tokens if t.features['norm'] not in sw]

    def _semfeat(self, i, dataset:Dataset):
        # incluir pos-tag?
        if not i.nextbool():
            for sentence in dataset.sentences:
                for token in sentence.tokens:
                    token.features.pop('pos')
                    token.features.pop('tag')

        # incluir dependencias
        if not i.nextbool():
            for sentence in dataset.sentences:
                for token in sentence.tokens:
                    token.features.pop('dep')

        self._umls(i, dataset)
        self._snomed(i, dataset)

        return dataset

    def _umls(self, i, dataset:Dataset):
        warnings.warn("UMLS not implemented yet")

        if i.nextbool():
            return True

    def _snomed(self, i, dataset:Dataset):
        warnings.warn("SNOMED not implemented yet")

        if i.nextbool():
            return True

    def _mulwords(self, i, dataset):
        #'MulWords' : 'countPhrase | freeling | all',
        choice = i.choose(self._collocations, self._postag, self._mulwnone)

        return choice(i, dataset)

    def _collocations(self, i, dataset:Dataset):
        warnings.warn("Collocation multiwords not implemented yet")

        tokens = [t.text for sentence in dataset.sentences for t in sentence.tokens]

        bigram_measures = nltk.collocations.BigramAssocMeasures()
        big_finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
        big_finder.apply_freq_filter(3)

        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        trig_finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)
        trig_finder.apply_freq_filter(3)

        bigrams = big_finder.nbest(bigram_measures.pmi, int(0.1 * len(tokens)))
        trigrams = trig_finder.nbest(trigram_measures.pmi, int(0.1 * len(tokens)))

        bigrams = set("_".join(b) for b in bigrams)
        trigrams = set("_".join(b) for b in trigrams)

        def filter(gram, features):
            return features['norm'] in bigrams or features['norm'] in trigrams

        return self._process_ngrams(dataset, 3, filter)

    def _postag(self, i, dataset:Dataset):
        ngram = i.nextint()

        def filter(gram, features):
            for t in gram:
                if t.features['pos'] not in ['NOUN', 'ADJ', 'ADP']:
                    return False

            if gram[0].features['pos'] != 'NOUN':
                return False

            if gram[-1].features['pos'] not in ['NOUN', 'ADJ']:
                return False

            return True

        return self._process_ngrams(dataset, ngram, filter)

    def _process_ngrams(self, dataset, ngram, filter):
        ids = max(k.id for sentence in dataset.sentences for k in sentence.tokens) * 10

        for sentence in dataset.sentences:
            ngrams = self._generate_ngrams(ngram, sentence.tokens)
            for gram in ngrams:
                features = dict(
                    norm="_".join(t.features['norm'] for t in gram),
                    pos="|".join(t.features['pos'] for t in gram),
                    tag="|".join(t.features['tag'] for t in gram),
                    dep="|".join(t.features['dep'] for t in gram),
                    vector=sum(t.features['vector'] for t in gram)
                )

                if not filter(gram, features):
                    continue

                start = gram[0].start
                end = gram[-1].end
                match = sentence.find_keyphrase(start=start, end=end)

                if match:
                    label = match.label
                    id = match.id
                else:
                    label = ''
                    id = ids
                    ids += 1

                phrase = Keyphrase(sentence, features, label, id, start, end)
                phrase.spacy_token = []

                for t in gram:
                    phrase.spacy_token.extend(t.spacy_token)

                sentence.tokens.append(phrase)

        dataset.max_length = max(len(s) for s in dataset.sentences)
        return dataset

    def _mulwnone(self, i, dataset:Dataset):
        return dataset

    def _generate_ngrams(self, ngrams, sequence:list):
        elements = []

        for n in range(2,ngrams+1):
            grams = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
            elements.extend(grams)

        return elements


    def _embed(self, i, dataset:Dataset):
        # 'Embed' : 'wordVec | onehot | none',
        choice = i.choose('wv', 'onehot', 'none')

        # train the dict-vectorizer in the relevant features
        feature_dicts = [dict(token.features) for sentence in dataset.sentences for token in sentence.tokens]

        for d in feature_dicts:
            d.pop('vector')
            d.pop('norm')

        print("Vectorizing...", end="")

        dvect = DictVectorizer(sparse=False).fit(feature_dicts)
        hashr = FeatureHasher(n_features=1000, input_type='string', non_negative=True)

        # now we vectorize
        for sentence in dataset.sentences:
            for token in sentence.tokens:
                # onehot encoding for words (hashing trick)
                text = token.features['norm']
                onehot = hashr.transform([text]).toarray().flatten()
                # word2vec encoding
                vector = token.features['vector']
                # semantic features
                semantic = dvect.transform([token.features]).flatten()

                # now maybe reuse that wv
                if choice == 'wv':
                    features = np.hstack((vector, semantic))
                elif choice == 'onehot':
                    features = np.hstack((onehot, semantic))
                else:
                    features = semantic

                token.features = features

        print("done")

        return dataset


    def _a(self, ind:Individual, dataset:Dataset):
        # choose between standard or sequence classifiers
        method = ind.choose(self._class, self._hmm)

        dataset = dataset.task_a()
        train, dev = dataset.split()
        prediction = method(ind, train, dev)

        prediction = (prediction > 0.5).reshape(-1).tolist()
        all_tokens = [token for sentence in dev.sentences for token in sentence.tokens]

        if 'cheat-a' in sys.argv:
            for token in all_tokens:
                token.mark_keyword(token.label != '')
            return
            # for s in dev.sentences:
            #     s.tokens = list(s.keyphrases)
            #     for token in s.tokens:
            #         token.mark_keyword(token.label != '')
            #         token.features = np.asarray([0])

        for token, is_kw in szip(all_tokens, prediction):
            token.mark_keyword(is_kw)

    def _b(self, ind:Individual, dataset:Dataset):
        dataset = dataset.task_b()
        train, dev = dataset.split()

        prediction = self._class(ind, train, dev)
        prediction = (prediction > 0.5).astype(int).reshape(-1).tolist()
        all_tokens = [token for sentence in dev.sentences for token in sentence.tokens if token.label != '']

        if 'cheat-b' in sys.argv:
            return

        for token, label in szip(all_tokens, prediction):
            token.mark_label(label)

    def _c(self, ind:Individual, dataset:Dataset):
        dataset = dataset.task_c()
        train, dev = dataset.split()

        prediction = self._class(ind, train, dev)
        prediction = prediction.astype(int)
        all_token_pairs = list(dev.token_pairs())

        if 'cheat-c' in sys.argv:
            for s in dev.sentences:
                s.predicted_relations = list(s.relations)
            return

        for (k1, k2), relations in szip(all_token_pairs, prediction):
            # print(relations)
            k1.sentence.add_predicted_relations(k1, k2, relations)

        # raise Exception()

    def _ab(self, ind, dataset):
        method = ind.choose(self._class, self._hmm)

        dataset = dataset.task_ab()
        train, dev = dataset.split()
        prediction = method(ind, train, dev)

        prediction = (prediction > 0.5).astype(int).reshape(-1).tolist()
        all_tokens = [token for sentence in dev.sentences for token in sentence.tokens]

        for token, label in szip(all_tokens, prediction):
            token.mark_ternary(label)

    def _bc(self, ind, dataset):
        dataset = dataset.task_bc()
        train, dev = dataset.split()

        prediction = self._class(ind, train, dev)
        prediction = (prediction > 0.5).astype(int)

        all_token_pairs = list(dev.token_pairs())

        for (k1, k2), relations in szip(all_token_pairs, prediction):
            relations, l1, l2 = np.split(relations, [6,7])
            k1.add_label_mark(l1[0])
            k2.add_label_mark(l2[0])

        for (k1, k2), relations in szip(all_token_pairs, prediction):
            relations, l1, l2 = np.split(relations, [6,7])
            k1.finish_label_mark()
            k2.finish_label_mark()
            k1.sentence.add_predicted_relations(k1, k2, relations)

    def _abc(self, ind, dataset):
        dataset = dataset.task_abc()
        train, dev = dataset.split()
        prediction = self._class(ind, train, dev)
        prediction = (prediction > 0.5).astype(int)

        all_token_pairs = list(dev.token_pairs())

        for (k1, k2), relations in szip(all_token_pairs, prediction):
            relations, kw1, l1, kw2, l2 = np.split(relations, [6,7,8,9])
            k1.add_keyword_mark(kw1[0])
            k1.add_label_mark(l1[0])
            k2.add_keyword_mark(kw2[0])
            k2.add_label_mark(l2[0])
            k1.sentence.add_predicted_relations(k1, k2, relations)

        for (k1, k2), relations in szip(all_token_pairs, prediction):
            relations, kw1, l1, kw2, l2 = np.split(relations, [6,7,8,9])
            k1.finish_keyword_mark()
            k2.finish_keyword_mark()
            k1.finish_label_mark()
            k2.finish_label_mark()
            k1.sentence.add_predicted_relations(k1, k2, relations)

    def _hmm(self, ind:Individual, train:Dataset, dev:Dataset):
        train_lengths = [len(s) for s in train.sentences]
        xtrain, ytrain = train.by_word()

        xdev, _ = dev.by_word()
        dev_lengths = [len(s) for s in dev.sentences]

        try:
            hmm = MultinomialHMM(decode=ind.choose('viterbi', 'bestfirst'),
                                 alpha=ind.nextfloat())
            hmm.fit(xtrain, ytrain, train_lengths)

            return hmm.predict(xdev, dev_lengths)

        except ValueError as e:
            if 'non-negative integers' in str(e):
                raise InvalidPipeline(str(e))
            elif 'unknown categories' in str(e):
                raise InvalidPipeline(str(e))
            else:
                raise

    def _crf(self, ind:Individual, xtrain, ytrain, xdev):
        raise NotImplementedError()

        crf = CRF()
        crf.fit(xtrain, ytrain)
        return [crf.predict(x) for x in xdev]

    def _class(self, ind:Individual, train:Dataset, dev:Dataset):
        #LR | nb | SVM | dt | NN
        des = ind.choose('lr', 'nb', 'svm', 'dt', 'nn')
        clss = None

        if des == 'lr':
            clss = self._lr(ind)
        elif des == 'nb':
            clss = MultinomialNB()
        elif des == 'svm':
            clss = self._svm(ind)
        elif des == 'dt':
            clss = DecisionTreeClassifier()
        else:
            return self._nn(ind, train, dev)

        xtrain, ytrain = train.by_word(balanced=False)

        # if len(ytrain.shape) > 1:
            # clss = OneVsRestClassifier(clss)

        clss.fit(xtrain, ytrain)

        xdev, ydev = dev.by_word()
        prediction = clss.predict(xdev)

        print("Validation score:", clss.score(xdev, ydev))
        print("Confusion matrix:")
        print(confusion_matrix(ydev, prediction))

        return prediction

    def _lr(self, i):
        lr = LogisticRegression(C=self._reg(i), penalty=self._penalty(i))
        return lr

    def _reg(self, i):
        return i.nextfloat()

    def _penalty(self, i):
        return i.choose('l1', 'l2')

    def _svm(self, i):
        return SVC(kernel=self._kernel(i))

    def _kernel(self, i):
        #linear | rbf | poly
        return i.choose('linear', 'rbf', 'poly')

    def _nn(self, ind:Individual, train:Dataset, dev:Dataset):
        try:
            # CVLayers DLayers FLayer Drop | RLayers DLayers FLayer Drop | DLayers FLayer Drop
            model = Sequential()
            option = ind.choose('conv', 'rec', 'deep')

            dropout = self._drop(ind)

            # The input representation depends on the kind of network
            if option == 'conv':
                xtrain, ytrain = train.by_sentence(balanced=False)
                xdev, _ = dev.by_sentence()

                x = Input(shape=xtrain[0].shape)
                y = self._cvlayers(ind, x, dropout)
                y = self._dlayers(ind, y, dropout)
                y = self._flayer(ind, y, train.predict_size, dropout)

            elif option == 'rec':
                xtrain, ytrain = train.by_sentence(balanced=False)
                xdev, _ = dev.by_sentence()

                x = Input(shape=xtrain[0].shape)
                y = self._rlayers(ind, x, dropout)
                y = self._dlayers(ind, y, dropout)
                y = self._flayer(ind, y, train.predict_size, dropout)

            else:
                xtrain, ytrain = train.by_word(balanced=False)
                xdev, _ = dev.by_word()

                x = Input(shape=xtrain[0].shape)
                y = self._dlayers(ind, x, dropout)
                y = self._flayer(ind, y, train.predict_size, dropout)

            model = Model(inputs=x, outputs=y)

            loss = 'binary_crossentropy'
            model.compile(optimizer='adam', loss=loss)

            is_categorical = train.predict_size != 1 and len(ytrain.shape) == 1

            if is_categorical:
                ytrain = to_categorical(ytrain)

            model.fit(xtrain, ytrain, validation_split=0.1)
            prediction = model.predict(xdev)

            if is_categorical:
                prediction = np.argmax(prediction, axis=1)

            return prediction

        except ValueError as e:
            msg = str(e)
            if 'out of bounds' in msg:
                raise InvalidPipeline('Bad NN architecture')
            if 'Negative dimension' in msg:
                raise InvalidPipeline('Bad NN architecture')
            else:
                raise e

    def _drop(self, i):
        return i.nextfloat()

    def _cvlayers(self, i, model, dropout):
        # 'CVLayers' : 'Count MinFilter MaxFilter FormatCon',
        count = self._count(i)
        minfilter = self._minfilter(i)
        maxfilter = max(minfilter, self._maxfilter(i))
        formatcon = self._formatcon(i)

        for _ in range(count):
            layers = []
            for kernel in range(minfilter, maxfilter + 1):
                if formatcon == 'all':
                    kernel_size = 2*kernel+1
                elif formatcon == 'same':
                    kernel_size = 2*minfilter+1
                else:
                    kernel_size = random.randint(2*minfilter+1, 2*maxfilter+2)

                layer = Conv1D(filters=1, kernel_size=kernel_size)(model)
                drop = Dropout(dropout)(layer)
                pool = MaxPooling1D(3)(drop)
                layers.append(pool)

            if len(layers) > 1:
                model = concatenate(layers)
            else:
                model = layers[0]

        return model

    def _count(self, i):
        return i.nextint() + 1

    def _minfilter(self, i):
        return i.nextint()

    def _maxfilter(self, i):
        return i.nextint()

    def _formatcon(self, i):
        return i.choose('same', 'all', 'rand')

    def _rlayers(self, i, model, dropout):
        # 'RLayers'  : 'Size',
        size = self._size(i)
        lstm = LSTM(size, dropout=dropout, recurrent_dropout=dropout)(model)
        return lstm

    def _size(self, i):
        return i.nextint()

    def _dlayers(self, i, model, dropout):
        # Dense layers
        # 'DLayers'  : 'Count MaxSize MinSize FormatDen Act',
        # 'Act'      : 'sigmoid | relu | tanh',
        # 'MinSize'  : 'i(10,100)',
        # 'MaxSize'  : 'i(10,100)',
        # #las capas van creciendo de tamaño del min al max, disminuyendo del max al min, todas del mismo tamaño
        # 'FormatDen': 'grow | shrink | same',

        count = self._count(i)
        minsize = self._minsize(i)
        maxsize = max(minsize, self._maxsize(i))
        activation = i.choose('sigmoid', 'relu', 'tanh')
        formatden = i.choose('grow', 'shrink', 'same')

        if formatden == 'grow':
            sizes = list(np.linspace(minsize, maxsize, count, dtype=int))
        elif formatden == 'shrink':
            sizes = list(np.linspace(maxsize, minsize, count, dtype=int))
        else:
            sizes = [minsize] * count

        for s in sizes:
            layer = Dense(s, activation=activation)(model)
            model = Dropout(dropout)(layer)

        return model

    def _minsize(self, i):
        return i.nextint()

    def _maxsize(self, i):
        return i.nextint()

    def _flayer(self, i, model, output_size, dropout):
        activation = 'sigmoid'
        z = Dense(output_size, activation=activation)(model)
        z = Dropout(dropout)(z)
        return z


def main():
    grammar = TassGrammar()
    print(grammar.size())
    return

    ge = PGE(grammar, verbose=True, popsize=100, selected=0.2, learning=0.05, errors='warn', timeout=300)
    ge.run(100)

    # for i in range(0, 100000):
    #     random.seed(i)

    #     ind = Individual([random.uniform(0,1) for _ in range(100)], grammar)
    #     sample = ind.sample()

    #     try:
    #         assert sample['Pipeline'][0]['Repr'][3]['SemFeat'][1]['Dep'] == ['yes']
    #         assert sample['Pipeline'][0]['Repr'][2]['MulWords'][0] == 'postag'
    #         assert sample['Pipeline'][0]['Repr'][5]['Embed'][0] == 'onehot'
    #         sample['Pipeline'][1]['A'][0]['Class'][0]['LR']
    #         sample['Pipeline'][2]['B'][0]['Class'][0]['LR']
    #         sample['Pipeline'][3]['C'][0]['Class'][0]['LR']
    #     except:
    #         continue

    #     print("\nRandom seed %i" % i)
    #     print(yaml.dump(sample))
    #     ind.reset()

    #     try:
    #         print(grammar.evaluate(ind))
    #         break
    #     except InvalidPipeline as e:
    #         print("Error", str(e))
    #         continue

if __name__ == '__main__':
    main()
