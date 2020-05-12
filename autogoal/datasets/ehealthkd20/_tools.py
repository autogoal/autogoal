# coding: utf8

import bisect
import sys


def offset(id):
    return id[0] + str(int(id[1:]) + 1000)


class EntityAnnotation:
    def __init__(self, id, typ, spans, text):
        self.id = id
        self.type = typ
        self.spans = spans
        self.text = text

    @staticmethod
    def parse(line):
        id, mid, text = line.strip().split("\t")
        typ, spans = mid.split(" ", 1)
        spans = [tuple(s.split()) for s in spans.split(";")]
        return EntityAnnotation(id, typ, spans, text)

    def __repr__(self):
        return "<Entity(id=%r, type=%r, spans=%r, text=%r)>" % (
            self.id,
            self.type,
            self.spans,
            self.text,
        )

    def offset_id(self):
        self.id = offset(self.id)

    def as_brat(self):
        spans = ";".join(" ".join(s) for s in self.spans)
        return "%s\t%s %s\t%s" % (self.id, self.type, spans, self.text)


class RelationAnnotation:
    def __init__(self, id, typ, arg1, arg2):
        self.id = id
        self.type = typ
        self.arg1 = arg1
        self.arg2 = arg2

    @staticmethod
    def parse(line):
        id, typ, arg1, arg2 = line.strip().split()
        arg1 = arg1.split(":")[1]
        arg2 = arg2.split(":")[1]
        return RelationAnnotation(id, typ, arg1, arg2)

    def offset_id(self):
        self.arg1 = offset(self.arg1)
        self.arg2 = offset(self.arg2)
        self.id = offset(self.id)

    def __repr__(self):
        return "<Relation(id=%r, type=%r, arg1=%r, arg2=%r)>" % (
            self.id,
            self.type,
            self.arg1,
            self.arg2,
        )

    def as_brat(self):
        return "%s\t%s Arg1:%s Arg2:%s" % (self.id, self.type, self.arg1, self.arg2)


class SameAsAnnotation:
    total = 0

    def __init__(self, id, typ, args):
        self.id = id
        self.type = typ
        self.args = args

    @staticmethod
    def parse(line):
        SameAsAnnotation.total += 1
        typ, args = line[1:].strip().split(" ", 1)
        id = "*%d" % SameAsAnnotation.total
        args = args.split()
        return SameAsAnnotation(id, typ, args)

    def offset_id(self):
        self.args = [offset(arg) for arg in self.args]

    def __repr__(self):
        return "<Relation(id=%r, type=%r, args=%r)>" % (self.id, self.type, self.args)

    def as_brat(self):
        return "*\t%s %s" % (self.type, " ".join(self.args))


class EventAnnotation:
    def __init__(self, id, typ, ref, args):
        self.id = id
        self.type = typ
        self.ref = ref
        self.args = args

    @staticmethod
    def parse(line):
        id, mid = line.strip().split("\t")
        args = mid.split()
        typ, ref = args[0].split(":")
        args = args[1:]
        args = {arg.split(":")[0]: arg.split(":")[1] for arg in args}
        return EventAnnotation(id, typ, ref, args)

    def offset_id(self):
        self.ref = offset(self.ref)
        self.id = offset(self.id)

        for k in self.args:
            self.args[k] = offset(self.args[k])

    def __repr__(self):
        return "<Event(id=%r, type=%r, ref=%r, args=%r)>" % (
            self.id,
            self.type,
            self.ref,
            self.args,
        )

    def as_brat(self):
        spans = " ".join(k + ":" + v for k, v in self.args.items())
        return "%s\t%s:%s %s" % (self.id, self.type, self.ref, spans)


class AttributeAnnotation:
    def __init__(self, id, typ, ref):
        self.id = id
        self.type = typ
        self.ref = ref

    @staticmethod
    def parse(line):
        id, typ, ref = line.strip().split()
        return AttributeAnnotation(id, typ, ref)

    def offset_id(self):
        self.ref = offset(self.ref)
        self.id = offset(self.id)

    def __repr__(self):
        return "<Attribute(id=%r, type=%r, ref=%r)>" % (self.id, self.type, self.ref)

    def as_brat(self):
        return "%s\t%s %s" % (self.id, self.type, self.ref)


class AnnFile:
    def __init__(self):
        self.annotations = []

    def load(self, path):
        with open(path) as fp:
            for line in fp:
                ann = self._parse(line)
                if ann:
                    self.annotations.append(ann)

        return self

    def annotations_of(self, type):
        for e in self.annotations:
            if isinstance(e, type):
                yield e

    def filter_sentences(self, sentences, order):
        skip = 0
        sentence = 0
        free_space = 0

        skipped_space = []
        selected_sentence_spans = []
        while order:
            next_sentence = order.pop(0) - 1

            skip_backup = skip
            while sentence != next_sentence:
                skip += len(sentences.pop(0)) + 1
                sentence += 1

            free_space += skip - skip_backup

            current_length = len(sentences[0])
            selected_sentence_spans.append((skip, skip + current_length))
            skipped_space.append(free_space)
            free_space -= current_length + 1

        selected_annotations = {}
        for entity in self.annotations_of(EntityAnnotation):
            min_start = min(int(start) for start, _ in entity.spans)
            max_end = max(int(end) for _, end in entity.spans)
            try:
                sentence = next(
                    i
                    for i, (start, end) in enumerate(selected_sentence_spans)
                    if (start <= min_start and max_end <= end)
                )
            except StopIteration:
                continue

            entity.spans = [
                tuple(str(int(x) - skipped_space[sentence]) for x in span)
                for span in entity.spans
            ]
            selected_annotations[entity.id] = entity

        for ann in self.annotations_of(EventAnnotation):
            if ann.ref in selected_annotations:
                selected_annotations[ann.id] = ann

        for ann in self.annotations:
            add = (
                isinstance(ann, SameAsAnnotation)
                and ann.args[0] in selected_annotations
            )
            add |= (
                isinstance(ann, RelationAnnotation) and ann.arg1 in selected_annotations
            )
            add |= (
                isinstance(ann, AttributeAnnotation) and ann.ref in selected_annotations
            )
            if add:
                selected_annotations[ann.id] = ann

        self.annotations = list(selected_annotations.values())

    def offset_spans(self, sentences, first):
        sentences_offset = self._compute_sentence_offset(sentences)

        for ann in self.annotations_of(EntityAnnotation):
            locations = list(
                set(
                    [
                        bisect.bisect_left(sentences_offset, int(s))
                        for span in ann.spans
                        for s in span
                    ]
                )
            )

            if len(locations) != 1:
                raise ValueError()

            location = locations.pop()
            offset = sentences_offset[location] + 1

            if first:
                offset = sentences_offset[location - 1] + 1 if location > 0 else 0

            ann.spans = [
                (str(int(span[0]) + offset), str(int(span[1]) + offset))
                for span in ann.spans
            ]

    def _compute_sentence_offset(self, sentences):
        sentences_offset = [-1]

        for s in sentences:
            prev = sentences_offset[-1]
            start = prev + 1
            end = start + len(s)
            sentences_offset.append(end)

        sentences_offset.pop(0)
        return sentences_offset

    def offset_ids(self):
        for ann in self.annotations:
            ann.offset_id()

    def _parse(self, line):
        if line.startswith("T"):
            return EntityAnnotation.parse(line)

        if line.startswith("R"):
            return RelationAnnotation.parse(line)

        if line.startswith("*"):
            return SameAsAnnotation.parse(line)

        if line.startswith("E"):
            return EventAnnotation.parse(line)

        if line.startswith("A"):
            return AttributeAnnotation.parse(line)

        if line.startswith("#"):
            return None

        raise ValueError("Unknown annotation: %s" % line)


def merge(ann1: str, ann2: str, text: str):
    """Merge annotations of two different versions of the same file.
    """
    file1 = AnnFile().load(ann1)
    file2 = AnnFile().load(ann2)
    sents = open(text).read().split("\n")

    file1.offset_spans(sents, first=True)
    file2.offset_spans(sents, first=False)
    file2.offset_ids()

    for ann in file1.annotations:
        print(ann.as_brat())

    for ann in file2.annotations:
        print(ann.as_brat())


def review(ann: str, text: str, order: str):
    """Process a merged annotation file and outputs the selected annotations.
    """
    file1 = AnnFile().load(ann)
    sents = open(text).read().split("\n")
    order = open(order).read().split("\n")
    order = [int(line.strip("*")) for line in order if line]

    file1.filter_sentences(sents, order)
    for ann in file1.annotations:
        print(ann.as_brat())


def review_text(text: str, order: str):
    """Process a merged annotation file and outputs the selected sentences.
    """
    sents = open(text).read().split("\n")
    order = open(order).read().split("\n")
    order = [int(line.strip("*")) for line in order if line]

    selected = [sents[i - 1] for i in order]
    for sent in selected:
        print(sent)


def to_review(order: str):
    order = open(order).read().split("\n")
    for i, o in enumerate(order):
        if o.endswith("*"):
            print(i + 1)
