# coding: utf-8

import pprint
import numpy as np
from scipy import sparse as sp
import bisect
import functools
from pathlib import Path
from ...utils import szip
from sklearn.feature_extraction import DictVectorizer
from typing import List
from tensorflow.keras.utils import to_categorical


all_relations = "is-a same-as part-of property-of subject target subject/target none".split()


def hstack(vectors):
    return np.hstack(vectors)


def vstack(vectors):
    return np.vstack(vectors)


def transform_relations(relations:set) -> int:
    if len(relations) > 2:
        raise ValueError("Invalid relations: {}".format(relations))

    if len(relations) == 0:
        return all_relations.index('none')

    if len(relations) == 1:
        return all_relations.index(list(relations)[0])

    if relations != {'subject', 'target'}:
        raise ValueError("Invalid relations: {}".format(relations))

    return all_relations.index('subject/target')


class Dataset:
    def __init__(self, sentences=None, validation_size=0, max_length=0):
        self.sentences = sentences or []
        self.validation_size = validation_size
        self.max_length = max_length

    def clone(self):
        return Dataset([s.clone() for s in self.sentences], self.validation_size)

    def by_word(self, balanced=False):
        raise NotImplementedError("Need to specialize dataset for a task first.")

    def by_sentence(self, balanced=False):
        raise NotImplementedError("Need to specialize dataset for a task first.")

    def __len__(self):
        return len(self.sentences)

    def load(self, finput:Path):
        goldA = finput.parent / ('output_A_' + finput.name[6:])
        goldB = finput.parent / ('output_B_' + finput.name[6:])
        goldC = finput.parent / ('output_C_' + finput.name[6:])

        text = finput.open().read()
        sentences = [s for s in text.split('\n') if s]

        self._parse_ann(sentences, goldA, goldB, goldC)

        return len(sentences)

    def _parse_ann(self, sentences, goldA, goldB, goldC):
        sentences_length = [len(s) for s in sentences]

        for i in range(1,len(sentences_length)):
            sentences_length[i] += (sentences_length[i-1] + 1)

        sentences_obj = [Sentence(text) for text in sentences]
        labels_by_id = {}
        sentence_by_id = {}

        for line in goldB.open():
            lid, lbl = line.split()
            labels_by_id[int(lid)] = lbl

        for line in goldA.open():
            lid, start, end = (int(i) for i in line.split())

            # find the sentence where this annotation is
            i = bisect.bisect(sentences_length, start)
            # correct the annotation spans
            if i > 0:
                start -= sentences_length[i-1] + 1
                end -= sentences_length[i-1] + 1

            # store the annotation in the corresponding sentence
            the_sentence = sentences_obj[i]
            the_sentence.keyphrases.append(Keyphrase(the_sentence, None, labels_by_id[lid], lid, start, end))
            sentence_by_id[lid] = the_sentence

        for line in goldC.open():
            rel, org, dest = line.split()
            org, dest = int(org), int(dest)

            # find the sentence this relation belongs ti
            the_sentence = sentence_by_id[org]
            assert the_sentence == sentence_by_id[dest]
            # and store it
            the_sentence.relations.append(Relation(the_sentence, org, dest, rel))

        for s in sentences_obj:
            s.sort()

        self.sentences.extend(sentences_obj)

    @property
    def feature_size(self):
        return self.sentences[0].tokens[0].features.shape[0]

    def split(self):
        train = Dataset(self.sentences[:-self.validation_size], max_length=self.max_length)
        dev = Dataset(self.sentences[-self.validation_size:], max_length=self.max_length)

        if self.__class__ != Dataset:
            train = self.__class__(train)
            dev = self.__class__(dev)

        return train, dev

    def token_pairs(self, enums=False):
        for sentence in self.sentences:
            for i, k1 in enumerate(sentence.tokens):
                # if k1.label == '':
                #     continue

                for j, k2 in enumerate(sentence.tokens):
                    # if k2.label == '':
                    #     continue

                    if enums:
                        yield i, j, k1, k2
                    else:
                        yield k1, k2

    def task_a(self):
        return TaskADataset(self)

    def task_b(self):
        return TaskBDataset(self)

    def task_c(self):
        return TaskCDataset(self)

    def task_ab(self):
        return TaskABDataset(self)

    def task_bc(self):
        return TaskBCDataset(self)

    def task_abc(self):
        return TaskABCDataset(self)

class TaskADataset(Dataset):
    def __init__(self, dataset:Dataset):
        self.sentences = dataset.sentences
        self.validation_size = dataset.validation_size
        self.max_length = dataset.max_length

    def clone(self):
        raise NotImplementedError("It is unsafe to clone a specialized dataset!")

    @property
    def predict_size(self):
        return 1

    def by_word(self, balanced=False):
        X = []
        y = []

        for sentence in self.sentences:
            for token in sentence.tokens:
                X.append(token.features)
                y.append(token.keyword_label)

        return vstack(X), hstack(y)

    def by_sentence(self, balanced=False):
        X = []
        y = []

        for sentence in self.sentences:
            for i, token in enumerate(sentence.tokens):
                X.append(sentence.token_features(self.max_length, i))
                y.append(token.keyword_label)

        return np.asarray(X), hstack(y)


class TaskBDataset(Dataset):
    def __init__(self, dataset:Dataset):
        self.sentences = dataset.sentences
        self.validation_size = dataset.validation_size
        self.max_length = dataset.max_length

    def clone(self):
        raise NotImplementedError("It is unsafe to clone a specialized dataset!")

    @property
    def predict_size(self):
        return 1

    def by_word(self, balanced=False):
        X = []
        y = []

        for sentence in self.sentences:
            for token in sentence.tokens:
                if token.label != '':
                    X.append(token.features)
                    y.append(token.binary_label)

        return vstack(X), hstack(y)

    def by_sentence(self, balanced=False):
        X = []
        y = []

        for sentence in self.sentences:
            for i, token in enumerate(sentence.tokens):
                if token.label != '':
                    X.append(sentence.token_features(self.max_length, i))
                    y.append(token.binary_label)

        return np.asarray(X), hstack(y)


class TaskCDataset(Dataset):
    def __init__(self, dataset:Dataset):
        self.sentences = dataset.sentences
        self.validation_size = dataset.validation_size
        self.max_length = dataset.max_length

    def clone(self):
        raise NotImplementedError("It is unsafe to clone a specialized dataset!")

    @property
    def predict_size(self):
        return 8

    def token_pairs(self, enums=False):
        for sentence in self.sentences:
            for i, k1 in enumerate(sentence.tokens):
                if k1.label == '':
                    continue

                for j, k2 in enumerate(sentence.tokens):
                    if k2.label == '':
                        continue

                    if enums:
                        yield i, j, k1, k2
                    else:
                        yield k1, k2

    def by_word(self, balanced=False):
        X = []
        y = []

        for k1, k2 in self.token_pairs():
            relations = k1.sentence.find_relations(k1.id, k2.id)
            relation_labels = {r.label for r in relations}
            relation_vector = transform_relations(relation_labels)

            joint_features = k1.sentence.joint_features(k1, k2)
            feature_vector = np.hstack((k1.features, k2.features, joint_features))

            X.append(feature_vector)
            y.append(relation_vector)

        return vstack(X), hstack(y)

    def by_sentence(self, balanced=False):
        X = []
        y = []

        for i, j, k1, k2 in self.token_pairs(enums=True):
            relations = k1.sentence.find_relations(k1.id, k2.id)
            relation_labels = {r.label for r in relations}
            relation_vector = transform_relations(relation_labels)

            if balanced and relation_vector.sum() == 0:
                continue

            feature_vector = k1.sentence.token_features(self.max_length, i, j)

            X.append(feature_vector)
            y.append(relation_vector)

        return np.asarray(X), hstack(y)


class TaskABDataset(Dataset):
    def __init__(self, dataset:Dataset):
        self.sentences = dataset.sentences
        self.validation_size = dataset.validation_size
        self.max_length = dataset.max_length

    def clone(self):
        raise NotImplementedError("It is unsafe to clone a specialized dataset!")

    @property
    def predict_size(self):
        return 3

    def by_word(self, balanced=False):
        X = []
        y = []

        for sentence in self.sentences:
            for token in sentence.tokens:
                X.append(token.features)
                y.append(token.ternary_label)

        return vstack(X), hstack(y)

    def by_sentence(self, balanced=False):
        X = []
        y = []

        for sentence in self.sentences:
            for i, token in enumerate(sentence.tokens):
                X.append(sentence.token_features(self.max_length, i))
                y.append(token.ternary_label)

        return np.asarray(X), hstack(y)


class TaskBCDataset(Dataset):
    def __init__(self, dataset:Dataset):
        self.sentences = dataset.sentences
        self.validation_size = dataset.validation_size
        self.max_length = dataset.max_length

    def clone(self):
        raise NotImplementedError("It is unsafe to clone a specialized dataset!")

    @property
    def predict_size(self):
        return 8

    def by_word(self, balanced=False):
        X = []
        y = []

        for k1, k2 in self.token_pairs():
            labels = np.asarray([k1.binary_label, k2.binary_label])

            relations = k1.sentence.find_relations(k1.id, k2.id)
            relation_labels = {r.label:True for r in relations}
            relation_vector = relation_mapper.transform([relation_labels]).toarray().reshape(-1)

            if balanced and relation_vector.sum() == 0:
                continue

            relation_vector = np.hstack((relation_vector, labels))
            feature_vector = np.hstack((k1.features, k2.features))

            X.append(feature_vector)
            y.append(relation_vector)

        return vstack(X), vstack(y)

    def by_sentence(self, balanced=False):
        X = []
        y = []

        for i, j, k1, k2 in self.token_pairs(enums=True):
            labels = np.asarray([k1.binary_label, k2.binary_label])

            relations = k1.sentence.find_relations(k1.id, k2.id)
            relation_labels = {r.label:True for r in relations}
            relation_vector = relation_mapper.transform([relation_labels]).toarray().reshape(-1)

            if balanced and relation_vector.sum() == 0:
                continue

            relation_vector = np.hstack((relation_vector, labels))
            feature_vector = k1.sentence.token_features(self.max_length, i, j)

            X.append(feature_vector)
            y.append(relation_vector)

        return np.asarray(X), vstack(y)


class TaskABCDataset(Dataset):
    def __init__(self, dataset:Dataset):
        self.sentences = dataset.sentences
        self.validation_size = dataset.validation_size
        self.max_length = dataset.max_length

    def clone(self):
        raise NotImplementedError("It is unsafe to clone a specialized dataset!")

    @property
    def predict_size(self):
        return 10

    def by_word(self, balanced=False):
        X = []
        y = []

        for k1, k2 in self.token_pairs():
            labels = np.asarray([k1.keyword_label, k1.binary_label, k2.keyword_label, k2.binary_label])

            relations = k1.sentence.find_relations(k1.id, k2.id)
            relation_labels = {r.label:True for r in relations}
            relation_vector = relation_mapper.transform([relation_labels]).toarray().reshape(-1)

            if balanced and relation_vector.sum() == 0:
                continue

            relation_vector = np.hstack((relation_vector, labels))
            feature_vector = np.hstack((k1.features, k2.features))

            X.append(feature_vector)
            y.append(relation_vector)

        return vstack(X), vstack(y)

    def by_sentence(self, balanced=False):
        X = []
        y = []

        for i, j, k1, k2 in self.token_pairs(enums=True):
            labels = np.asarray([k1.keyword_label, k1.binary_label, k2.keyword_label, k2.binary_label])

            relations = k1.sentence.find_relations(k1.id, k2.id)
            relation_labels = {r.label:True for r in relations}
            relation_vector = relation_mapper.transform([relation_labels]).toarray().reshape(-1)

            if balanced and relation_vector.sum() == 0:
                continue

            relation_vector = np.hstack((relation_vector, labels))
            feature_vector = k1.sentence.token_features(self.max_length, i, j)

            X.append(feature_vector)
            y.append(relation_vector)

        return np.asarray(X), vstack(y)


class Keyphrase:
    def __init__(self, sentence, features, label, id, start, end):
        self.sentence = sentence
        self.features = features
        self.label = label
        self.id = id
        self.start = start
        self.end = end
        self.spacy_token = None

        self._all_keywords = []
        self._all_labels = []

    def clone(self, sentence):
        return Keyphrase(sentence, self.features, self.label, self.id, self.start, self.end)

    @property
    def text(self) -> str:
        return self.sentence.text[self.start:self.end]

    @property
    def keyword_label(self) -> int:
        return 0 if self.label == '' else 1

    @property
    def binary_label(self) -> int:
        return 0 if self.label == 'Action' else 1

    @property
    def ternary_label(self) -> int:
        return 0 if self.label == '' else 1 if self.label == 'Action' else 2

    def mark_keyword(self, value):
        if hasattr(self, 'is_kw'):
            raise ValueError("Already marked!")

        self.is_kw = bool(value)

    def add_keyword_mark(self, value):
        if hasattr(self, 'is_kw'):
            raise ValueError("Already marked!")

        self._all_keywords.append(int(value))

    def finish_keyword_mark(self):
        if not self._all_keywords:
            return

        self.mark_keyword(np.mean(self._all_keywords) > 0.5)
        self._all_keywords.clear()

    def mark_label(self, value):
        if not hasattr(self, 'is_kw'):
            raise ValueError("Must be marked as keyword first")

        if isinstance(value, int):
            value = ['Action', 'Concept'][value]

        self.label = value if self.is_kw else ''

    def add_label_mark(self, value):
        self._all_labels.append(int(value))

    def finish_label_mark(self):
        if not self._all_labels:
            return

        self.mark_label(int(np.mean(self._all_labels)) > 0.5)
        self._all_labels.clear()

    def mark_ternary(self, value):
        if hasattr(self, 'is_kw'):
            raise ValueError("Already marked!")

        # if isinstance(value, np.ndarray):
        #     value = np.argmax(a)

        if isinstance(value, int):
            value = ['', 'Action', 'Concept'][value]

        self.label = value
        self.is_kw = self.label != ''

    def __repr__(self):
        return "Keyphrase(text=%r, label=%r, id=%r)" % (self.text, self.label, self.id)


class Relation:
    def __init__(self, sentence, origin, destination, label):
        self.sentence = sentence
        self.origin = origin
        self.destination = destination
        self.label = label

    def clone(self, sentence):
        return Relation(sentence, self.origin, self.destination, self.label)

    @property
    def from_phrase(self) -> Keyphrase:
        return self.sentence.find_keyphrase(id=self.origin)

    @property
    def to_phrase(self) -> Keyphrase:
        return self.sentence.find_keyphrase(id=self.destination)

    class _Unk:
        text = 'UNK'

    def __repr__(self):
        from_phrase = (self.from_phrase or Relation._Unk()).text
        to_phrase = (self.to_phrase or Relation._Unk()).text
        return "Relation(from=%r, to=%r, label=%r)" % (from_phrase, to_phrase, self.label)


class Sentence:
    def __init__(self, text:str):
        self.text = text
        self.keyphrases = []
        self.relations = []
        self.tokens = []
        self.predicted_relations = []

    def clone(self):
        s = Sentence(self.text)
        s.keyphrases = [k.clone(s) for k in self.keyphrases]
        s.relations = [r.clone(s) for r in self.relations]
        s.tokens = [k.clone(s) for k in self.tokens]
        return s

    def find_keyphrase(self, id=None, start=None, end=None) -> Keyphrase:
        if id is not None:
            return self._find_keyphrase_by_id(id)
        return self._find_keyphrase_by_spans(start, end)

    def find_relations(self, orig, dest):
        results = []

        for r in self.relations:
            if r.origin == orig and r.destination == dest:
                results.append(r)

        return results

    def find_relation(self, orig, dest, label):
        for r in self.relations:
            if r.origin == orig and r.destination == dest and label == r.label:
                return r

        return None

    def joint_features(self, k1, k2):
        # relations = {r.label for r in self.find_relations(k1.id, k2.id)}

        i1 = np.mean([t.i for t in k1.spacy_token])
        i2 = np.mean([t.i for t in k2.spacy_token])
        distance = i1 - i2

        xa,ya = k1.start, k1.end
        xb,yb = k2.start, k2.end

        contained = int(xa >= xb and ya <= yb)

        # k1.spacy_token

        # parent = k1.spacy_token[0]

        # while not parent.is_ancestor(k2.spacy_token[0]):
        #     parent = parent.head

        return np.asarray([distance, contained])

    def _find_keyphrase_by_id(self, id):
        for k in self.keyphrases:
            if k.id == id:
                return k

        return None

    def _find_keyphrase_by_spans(self, start, end):
        for k in self.keyphrases:
            if k.start == start and k.end == end:
                return k

        return None

    def token_features(self, max_length:int, *index:int):
        X = []

        for token in self.tokens:
            X.append(token.features)

        X = np.asarray(X)
        padding = max_length - len(X)

        if padding > 0:
            _, cols = X.shape
            X = np.vstack((X, np.zeros((padding, cols))))

        idxcols = np.zeros((max_length, len(index)))
        for col,row in enumerate(index):
            idxcols[row, col] = 1.0

        return np.hstack((X, idxcols))

    def add_predicted_relations(self, k1, k2, relations):
        if not isinstance(relations, list):
            relations = all_relations[relations].split('/')

        if relations[0] == 'none':
            relations = []

        # print("!!!!!!!!", relations, k1.label, k1.id, k2.label, k2.id)

        for relation in relations:
            if relation in ['subject', 'target']:
                if k1.label != 'Action' or k2.label != 'Concept':
                    continue
            else:
                if k1.label != 'Concept' or k2.label != 'Concept':
                    continue

            self.predicted_relations.append(Relation(self, k1.id, k2.id, relation))

    def invert(self):
        s = Sentence(self.text)
        s.keyphrases = [t for t in self.tokens if t.is_kw]
        s.relations = self.predicted_relations
        return s

    def sort(self):
        self.keyphrases.sort(key=lambda k: (k.start, k.end))

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return "Sentence(text=%r, keyphrases=%r, relations=%r)" % (self.text, self.keyphrases, self.relations)
