# coding: utf8

from .data import get_data_for


class DocToken2Sent:
    def train(self, X, y=None):
        return self.run(X)

    def run(self, X):
        sents = []

        for doc in X:
            sents.extend(doc.tokens)

        return sents


def build_ontology_adapters(onto):
    OntoML = onto.Software("OntoML")

    DocToken2SentAdaptor = onto.Adaptor("DocToken2SentAdaptor")
    DocToken2SentAdaptor.implementedIn = OntoML
    DocToken2SentAdaptor.importCode = "ontoml.adapters.DocToken2Sent"
    DocToken2SentAdaptor.hasInput = get_data_for(onto.DocumentCorpus, onto.Tokenized)
    DocToken2SentAdaptor.hasOutput = get_data_for(onto.SentenceCorpus)

