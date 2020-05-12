# coding: utf8

import bisect
import collections
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List

from autogoal.datasets.ehealthkd20._tools import (
    AnnFile,
    AttributeAnnotation,
    EntityAnnotation,
    EventAnnotation,
    RelationAnnotation,
    SameAsAnnotation,
)

ENTITIES = ["Concept", "Action", "Predicate", "Reference"]

RELATIONS = [
    "is-a",
    "same-as",
    "part-of",
    "has-property",
    "causes",
    "entails",
    "in-context",
    "in-place",
    "in-time",
    "subject",
    "target",
    "domain",
    "arg",
]


class Keyphrase:
    def __init__(self, sentence, label, id, spans):
        self.sentence: Sentence = sentence
        self.label = label
        self.id = id
        self.spans = spans
        self.attributes: List[Attribute] = []

    def split(self):
        if len(self.spans) > 1:
            raise TypeError("Cannot split a keyphrase with multiple spans")

        start, end = self.spans[0]
        spans = []
        spans.append(start)

        for i, c in enumerate(self.text):
            if c == " ":
                spans.append(start + i)
                spans.append(start + i + 1)

        spans.append(end)
        self.spans = [(spans[i], spans[i + 1]) for i in range(0, len(spans), 2)]

    def clone(self, sentence, shallow=False) -> "Keyphrase":
        k = Keyphrase(sentence, self.label, self.id, self.spans)
        k.attributes = [a if shallow else a.clone(k) for a in self.attributes]
        return k

    @property
    def text(self):
        return " ".join(self.sentence.text[s:e] for (s, e) in self.spans)

    def __repr__(self):
        return "Keyphrase(text=%r, label=%r, id=%r, attr=%r)" % (
            self.text,
            self.label,
            self.id,
            self.attributes,
        )

    def as_ann(self, shift):
        return "T{0}\t{1} {2}\t{3}\n".format(
            self.id,
            self.label,
            ";".join(
                "{} {}".format(start + shift, end + shift) for start, end in self.spans
            ),
            self.text,
        )

    def matches(self, other: "Keyphrase", label=None):
        return (
            isinstance(other, Keyphrase)
            and self.sentence.text == other.sentence.text
            and self.spans == other.spans
            and (
                (label is None and self.label == other.label)
                or (label is not None and self.label == label)
            )
        )

    def find_attributes(self, label) -> "Attribute":
        return [attr for attr in self.attributes if attr.label == label]


class Relation:
    def __init__(self, sentence, origin, destination, label):
        self.sentence = sentence
        self.origin = origin
        self.destination = destination
        self.label = label

    def clone(self, sentence) -> "Relation":
        return Relation(sentence, self.origin, self.destination, self.label)

    @property
    def from_phrase(self) -> Keyphrase:
        return self.sentence.find_keyphrase(id=self.origin)

    @property
    def to_phrase(self) -> Keyphrase:
        return self.sentence.find_keyphrase(id=self.destination)

    class _Unk:
        text = "UNK"

    def __repr__(self):
        from_phrase = (self.from_phrase or Relation._Unk()).text
        to_phrase = (self.to_phrase or Relation._Unk()).text
        return "Relation(from=%r, to=%r, label=%r)" % (
            from_phrase,
            to_phrase,
            self.label,
        )

    def as_ann(self, shift):
        if self.label == "same-as":
            return "*\tsame-as T{0} T{1}\n".format(self.origin, self.destination)
        else:
            return "R{0}\t{1} Arg1:T{2} Arg2:T{3}\n".format(
                shift, self.label, self.origin, self.destination
            )

    def matches(self, other: "Relation", label=None):
        return (
            isinstance(other, Relation)
            and self.sentence.text == other.sentence.text
            and self.from_phrase.matches(other.from_phrase)
            and self.to_phrase.matches(other.to_phrase)
            and (
                (label is None and self.label == other.label)
                or (label is not None and self.label == label)
            )
        )


class Attribute:
    def __init__(self, keyphrase: Keyphrase, label):
        self.keyphrase = keyphrase
        self.label = label

    def clone(self, keyphrase) -> "Attribute":
        return Attribute(keyphrase, self.label)

    def __repr__(self):
        return "Attribute(label=%r)" % (self.label,)

    def as_ann(self, shift):
        return "A{0}\t{1} T{2}\n".format(shift, self.label, self.keyphrase.id)


class Sentence:
    def __init__(self, text):
        self.text = text
        self.keyphrases: List[Keyphrase] = []
        self.relations: List[Relation] = []

    def clone(self, shallow=False) -> "Sentence":
        s = Sentence(self.text)
        s.keyphrases = [k if shallow else k.clone(s) for k in self.keyphrases]
        s.relations = [r if shallow else r.clone(s) for r in self.relations]
        return s

    def fix_ids(self, start=1):
        next_id = start

        copy = self.clone()
        for k, kc in zip(self.keyphrases, copy.keyphrases):
            for r, rc in zip(self.relations, copy.relations):
                if rc.origin == kc.id:
                    r.origin = next_id
                if rc.destination == kc.id:
                    r.destination = next_id

            k.id = next_id
            next_id += 1

        return next_id

    def overlapping_keyphrases(self):
        result = []

        for s1 in self.keyphrases:
            overlaps = set([s1])

            for s2 in self.keyphrases:
                if s2.spans == s1.spans:
                    overlaps.add(s2)

            if len(overlaps) > 1 and overlaps not in result:
                result.append(overlaps)

        return result

    def merge_overlapping_keyphrases(self):
        overlaps = self.overlapping_keyphrases()

        for keyphrases in overlaps:
            keyphrases = list(keyphrases)
            first = keyphrases[0]
            rest = keyphrases[1:]
            rest_ids = [k.id for k in rest]

            for relation in self.relations:
                if relation.origin in rest_ids:
                    print(
                        "Changing %r origin from %s to %s"
                        % (relation, relation.origin, first.id)
                    )
                    relation.origin = first.id
                if relation.destination in rest_ids:
                    print(
                        "Changing %r destination from %s to %s"
                        % (relation, relation.destination, first.id)
                    )
                    relation.destination = first.id

            for keyp in rest:
                self.keyphrases.remove(keyp)

    def dup_relations(self):
        dup_relations = collections.defaultdict(lambda: [])

        for r in self.relations:
            dup_relations[(r.label, r.origin, r.destination)].append(r)

        return {k: v for k, v in dup_relations.items() if len(v) > 1}

    def remove_dup_relations(self):
        new_relations = {}

        for r in self.relations:
            new_relations[(r.label, r.origin, r.destination)] = r

        self.relations = list(new_relations.values())

    def find_first_match(self, annotation, label=None):
        matches = self.find_matches(annotation, label)
        return None if not matches else matches[0]

    def find_matches(self, annotation, label=None):
        if isinstance(annotation, Keyphrase):
            return [k for k in self.keyphrases if k.matches(annotation, label)]
        elif isinstance(annotation, Relation):
            return [r for r in self.relations if r.matches(annotation, label)]
        else:
            raise TypeError("Invalid annotation")

    def find_keyphrase(self, id=None, start=None, end=None, spans=None) -> Keyphrase:
        if id is not None:
            return self._find_keyphrase_by_id(id)
        if spans is None:
            spans = [(start, end)]
        return self._find_keyphrase_by_spans(spans)

    def find_relations(self, orig, dest) -> List[Relation]:
        results = []

        for r in self.relations:
            if r.origin == orig and r.destination == dest:
                results.append(r)

        return results

    def find_relation(self, orig, dest, label) -> Relation:
        for r in self.relations:
            if r.origin == orig and r.destination == dest and label == r.label:
                return r

        return None

    def _find_keyphrase_by_id(self, id) -> Keyphrase:
        for k in self.keyphrases:
            if k.id == id:
                return k

        return None

    def _find_keyphrase_by_spans(self, spans) -> Keyphrase:
        for k in self.keyphrases:
            if k.spans == spans:
                return k

        return None

    def sort(self):
        self.keyphrases.sort(
            key=lambda k: tuple([s for s, e in k.spans] + [e for s, e in k.spans])
        )

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        return "Sentence(text=%r, keyphrases=%r, relations=%r)" % (
            self.text,
            self.keyphrases,
            self.relations,
        )

    @staticmethod
    def load(finput) -> "List[Sentence]":
        return [
            Sentence(s.strip())
            for s in finput.read_text(encoding="utf8").splitlines()
            if s
        ]

    @property
    def annotated(self):
        return self.keyphrases or self.relations


class Collection:
    def __init__(self, sentences=None):
        self.sentences: List[Sentence] = sentences or []

    def clone(self, skip_empty=False) -> "Collection":
        return Collection(
            [s.clone() for s in self.sentences if not skip_empty or s.annotated]
        )

    def merge(self, *collections: "Collection", skip_empty=False):
        clone = self.clone(skip_empty)
        sentences = [
            s.clone()
            for c in collections
            for s in c.sentences
            if not skip_empty or s.annotated
        ]
        clone.sentences.extend(sentences)
        return clone

    def __len__(self):
        return len(self.sentences)

    def fix_ids(self):
        next_id = 1

        for s in self.sentences:
            next_id = s.fix_ids(next_id)

    def filter(self, keyphrase=None, relation=None, attribute=None,) -> "Collection":
        sentences = []
        for sentence in self.sentences:
            s = Sentence(sentence.text)

            # keyphrases
            s.keyphrases = [
                k.clone(s)
                for k in sentence.keyphrases
                if keyphrase is None or keyphrase(k)
            ]

            # attributes
            if attribute is not None:
                for k in s.keyphrases:
                    k.attributes = [
                        a for a in k.attributes if attribute is None or attribute(a)
                    ]
                s.keyphrases = [k for k in s.keyphrases if k.attributes]

            # relations
            s.relations = [
                r.clone(s)
                for r in sentence.relations
                if (relation is None or relation(r))
                and (
                    keyphrase is None
                    or (keyphrase(r.from_phrase) and keyphrase(r.to_phrase))
                )
                and (
                    attribute is None
                    or (
                        any(attribute(a) for a in r.from_phrase.attributes)
                        and any(attribute(a) for a in r.to_phrase.attributes)
                    )
                )
            ]
            sentences.append(s)
        return Collection(sentences)

    def filter_keyphrase(self, labels) -> "Collection":
        return self.filter(keyphrase=lambda k: k.label in labels)

    def filter_relation(self, labels) -> "Collection":
        return self.filter(relation=lambda r: r.label in labels)

    def filter_attribute(self, labels) -> "Collection":
        return self.filter(attribute=lambda a: a.label in labels)

    def find_first_match(self, text) -> Sentence:
        matches = self.find_matches(text)
        return None if not matches else matches[0]

    def find_matches(self, text) -> List[Sentence]:
        return [s for s in self.sentences if s.text == text]

    def load(
        self,
        finput: Path,
        *,
        legacy=True,
        keyphrases=True,
        relations=True,
        attributes=True
    ) -> "Collection":
        return CollectionV2Handler.load(
            self,
            finput,
            legacy=legacy,
            keyphrases=keyphrases,
            relations=relations,
            attributes=attributes,
        )

    def dump(self, text_file: Path, skip_empty_sentences=True):
        return CollectionV2Handler.dump(self, text_file, skip_empty_sentences)

    def load_dir(
        self,
        finput: Path,
        *,
        legacy=True,
        keyphrases=True,
        relations=True,
        attributes=True
    ) -> "Collection":
        return CollectionV2Handler.load_dir(
            self,
            finput,
            legacy=legacy,
            keyphrases=keyphrases,
            relations=relations,
            attributes=attributes,
        )


class CollectionHandler:
    @classmethod
    def load_dir(cls, collection: Collection, finput: Path, **kargs) -> Collection:
        pass

    @classmethod
    def load(cls, collection: Collection, finput: Path, **kargs) -> Collection:
        pass

    @classmethod
    def dump(cls, collection: Collection, text_file: Path, skip_empty_sentences=True):
        pass


class CollectionV1Handler(CollectionHandler):
    @classmethod
    def load_dir(cls, collection: Collection, finput: Path) -> Collection:
        for item in finput.iterdir():
            if re.fullmatch(r".*put_scenario.*\.txt", item.name):
                cls.load(collection, item)
        return collection

    @classmethod
    def load(cls, collection: Collection, finput: Path) -> Collection:
        input_b_file = finput.parent / ("output_b_" + finput.name.split("_")[1])

        sentence_by_id = cls._load_keyphrases(collection, finput)

        for line in input_b_file.open(encoding="utf8").readlines():
            label, src, dst = line.strip().split("\t")
            src, dst = int(src), int(dst)

            the_sentence = sentence_by_id[src]

            if the_sentence != sentence_by_id[dst]:
                warnings.warn(
                    "In file '%s' relation '%s' between %i and %i crosses sentence boundaries and has been ignored."
                    % (finput, label, src, dst)
                )
                continue

            assert sentence_by_id[dst] == the_sentence

            the_sentence.relations.append(
                Relation(the_sentence, src, dst, label.lower())
            )

        return collection

    @classmethod
    def _load_keyphrases(cls, collection: Collection, finput: Path):
        cls._load_input(collection, finput)

        input_a_file = finput.parent / ("output_a_" + finput.name.split("_")[1])

        sentences_length = [len(s.text) for s in collection.sentences]
        for i in range(1, len(sentences_length)):
            sentences_length[i] += sentences_length[i - 1] + 1

        sentence_by_id = {}

        for line in input_a_file.open(encoding="utf8").readlines():
            lid, spans, label, _ = line.strip().split("\t")
            lid = int(lid)

            spans = [s.split() for s in spans.split(";")]
            spans = [(int(start), int(end)) for start, end in spans]

            # find the sentence where this annotation is
            i = bisect.bisect(sentences_length, spans[0][0])
            # correct the annotation spans
            if i > 0:
                spans = [
                    (
                        start - sentences_length[i - 1] - 1,
                        end - sentences_length[i - 1] - 1,
                    )
                    for start, end in spans
                ]
                spans.sort(key=lambda t: t[0])
            # store the annotation in the corresponding sentence
            the_sentence = collection.sentences[i]
            keyphrase = Keyphrase(the_sentence, label, lid, spans)
            the_sentence.keyphrases.append(keyphrase)

            if len(keyphrase.spans) == 1:
                keyphrase.split()

            sentence_by_id[lid] = the_sentence

        return sentence_by_id

    @classmethod
    def _load_input(cls, collection: Collection, finput: Path):
        sentences = [s.strip() for s in finput.open(encoding="utf8").readlines() if s]
        sentences_obj = [Sentence(text) for text in sentences]
        collection.sentences.extend(sentences_obj)

    @classmethod
    def dump(cls, collection: Collection, text_file: Path, skip_empty_sentences=True):
        collection.fix_ids()

        input_file = text_file.open("w", encoding="utf8")
        output_a_file = (
            text_file.parent / ("output_a_" + text_file.name.split("_")[1])
        ).open("w", encoding="utf8")
        output_b_file = (
            text_file.parent / ("output_b_" + text_file.name.split("_")[1])
        ).open("w", encoding="utf8")

        shift = 0

        for sentence in collection.sentences:
            if (
                not sentence.keyphrases
                and not sentence.relations
                and skip_empty_sentences
            ):
                continue

            input_file.write("{}\n".format(sentence.text))

            for keyphrase in sentence.keyphrases:
                output_a_file.write(
                    "{0}\t{1}\t{2}\t{3}\n".format(
                        keyphrase.id,
                        ";".join(
                            "{} {}".format(start + shift, end + shift)
                            for start, end in keyphrase.spans
                        ),
                        keyphrase.label,
                        keyphrase.text,
                    )
                )

            for relation in sentence.relations:
                output_b_file.write(
                    "{0}\t{1}\t{2}\n".format(
                        relation.label, relation.origin, relation.destination
                    )
                )

            shift += len(sentence) + 1


class CollectionV2Handler(CollectionHandler):
    @classmethod
    def load_dir(
        cls,
        collection: Collection,
        finput: Path,
        *,
        legacy=True,
        keyphrases=True,
        relations=True,
        attributes=True
    ) -> Collection:
        for item in finput.iterdir():
            if item.suffix == ".txt":
                cls.load(
                    collection,
                    item,
                    legacy=legacy,
                    keyphrases=keyphrases,
                    relations=relations,
                    attributes=attributes,
                )

        return collection

    @classmethod
    def load(
        cls,
        collection: Collection,
        finput: Path,
        *,
        legacy=True,
        keyphrases=True,
        relations=True,
        attributes=True
    ) -> "Collection":

        # add sentences from input .txt to Collection
        sentences = cls._load_input(collection, finput)

        # if keyphrases won't be loaded finish right there
        if not keyphrases:
            return collection

        # else, parse .ann file to start the annotation of sentences
        ann_file = cls._load_ann(finput)

        def add_relation(source_id, destination_id, ann_type, id_to_keyphrase):
            source = id_to_keyphrase[source_id]
            destination = id_to_keyphrase[destination_id]
            if source.sentence != destination.sentence:
                warnings.warn(
                    "In file '%s' relation '%s' between %i and %i crosses sentence boundaries and has been ignored."
                    % (finput, ann_type, source_id, destination_id)
                )
            else:
                relation = Relation(
                    source.sentence, source.id, destination.id, ann_type
                )
                source.sentence.relations.append(relation)

        def legacy_load(ann_file, sentences, id_to_keyphrase):
            for ann in ann_file.annotations:
                if isinstance(ann, EventAnnotation):
                    id_to_keyphrase[ann.id] = id_to_keyphrase[ann.ref]

            for ann in ann_file.annotations:
                if not isinstance(ann, EventAnnotation):
                    continue
                for label, destination in ann.args.items():
                    label = "".join(i for i in label if not i.isdigit()).lower()
                    add_relation(ann.ref, destination, label, id_to_keyphrase)

        # compute sentences' boundaries
        sentences_length = [len(s) for s in sentences]
        for i in range(1, len(sentences_length)):
            sentences_length[i] += sentences_length[i - 1] + 1

        # load keyphrases from Entity Annotations
        id_to_keyphrase = {}
        for ann in ann_file.annotations:
            if isinstance(ann, EntityAnnotation):
                tid = int(ann.id[1:])
                spans = [(int(start), int(end)) for start, end in ann.spans]
                sid, spans = cls._get_relative_ann(spans, sentences_length)
                sentence = sentences[sid]
                keyphrase = Keyphrase(sentence, ann.type, tid, spans)
                sentence.keyphrases.append(keyphrase)
                if len(keyphrase.spans) == 1:
                    keyphrase.split()
                id_to_keyphrase[ann.id] = keyphrase

        # load relations from Event Annotations (legacy support)
        if legacy and relations:
            legacy_load(ann_file, sentences, id_to_keyphrase)

        # load standard relations and attributes
        for ann in ann_file.annotations:
            if isinstance(ann, RelationAnnotation) and relations:
                add_relation(ann.arg1, ann.arg2, ann.type, id_to_keyphrase)

            elif isinstance(ann, SameAsAnnotation) and relations:
                source = ann.args[0]
                for destination in ann.args[1:]:
                    add_relation(source, destination, ann.type, id_to_keyphrase)

            elif isinstance(ann, AttributeAnnotation) and attributes:
                keyphrase = id_to_keyphrase[ann.ref]
                attribute = Attribute(keyphrase, ann.type)
                keyphrase.attributes.append(attribute)

            elif not (
                isinstance(ann, EntityAnnotation)
                or legacy
                and isinstance(ann, EventAnnotation)
            ):
                warnings.warn(
                    "In file '%s' annotation '%s' has been ignored." % (finput, ann)
                )

        for s in sentences:
            s.sort()
        return collection

    @classmethod
    def _load_input(cls, collection: Collection, finput: Path) -> List[Sentence]:
        sentences = Sentence.load(finput)
        collection.sentences.extend(sentences)
        return sentences

    @classmethod
    def _load_ann(cls, finput: Path) -> AnnFile:
        ann_path: Path = finput.parent / (finput.stem + ".ann")
        return AnnFile().load(ann_path)

    @classmethod
    def _get_relative_ann(cls, spans, sentences_length: List[int]) -> int:
        # find the sentence where this annotation is
        i = bisect.bisect(sentences_length, spans[0][0])
        # correct the annotation spans
        if i > 0:
            spans = [
                (
                    start - sentences_length[i - 1] - 1,
                    end - sentences_length[i - 1] - 1,
                )
                for start, end in spans
            ]
            spans.sort(key=lambda t: t[0])
        return i, spans

    @classmethod
    def dump(cls, collection: Collection, text_file, skip_empty_sentences=True):
        ann_path: Path = text_file.parent / (text_file.stem + ".ann")
        cls._dump_input(collection, text_file, skip_empty_sentences)
        cls._dump_ann(collection, ann_path, skip_empty_sentences)

    @classmethod
    def _dump_input(
        cls, collection: Collection, text_file: Path, skip_empty_sentences=True
    ):
        text_file.parent.mkdir(parents=True, exist_ok=True)
        text_file.write_text(
            "\n".join(
                sentence.text
                for sentence in collection.sentences
                if not skip_empty_sentences or sentence.keyphrases or sentence.relations
            ),
            encoding="utf8",
        )

    @classmethod
    def _dump_ann(
        cls, collection: Collection, ann_path: Path, skip_empty_sentences=True
    ):
        collection.fix_ids()

        aid = 0
        rid = 0
        shift = 0
        with ann_path.open("w", encoding="utf8") as ann_file:
            for sentence in collection.sentences:
                if (
                    skip_empty_sentences
                    and not sentence.keyphrases
                    and not sentence.relations
                ):
                    continue

                for keyphrase in sentence.keyphrases:
                    ann_file.write(keyphrase.as_ann(shift))

                    for attribute in keyphrase.attributes:
                        ann_file.write(attribute.as_ann(aid))
                        aid += 1

                for relation in sentence.relations:
                    ann_file.write(relation.as_ann(rid))
                    if relation.label != "same-as":
                        rid += 1

                shift += len(sentence) + 1


class DisjointSet:
    def __init__(self, *items):
        self.nodes = {x: DisjointNode(x) for x in items}

    def merge(self, items):
        items = (self.nodes[x] for x in items)
        try:
            head, *others = items
            for other in others:
                head.merge(other)
        except ValueError:
            pass

    @property
    def representatives(self):
        return {n.representative for n in self.nodes.values()}

    @property
    def groups(self):
        return [
            [n for n in self.nodes.values() if n.representative == r]
            for r in self.representatives
        ]

    def __len__(self):
        return len(self.representatives)

    def __getitem__(self, item):
        return self.nodes[item]

    def __call__(self, item1, item2):
        return self[item1].representative == self[item2].representative

    def __str__(self):
        return str(self.groups)

    def __repr__(self):
        return str(self)


class DisjointNode:
    def __init__(self, value):
        self.value = value
        self.parent = self

    @property
    def representative(self):
        if self.parent != self:
            self.parent = self.parent.representative
        return self.parent

    def merge(self, other):
        other.representative.parent = self.representative

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)
