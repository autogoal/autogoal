from autogoal.datasets.ehealthkd20._utils import Collection, Sentence


def to_biluov(tokensxsentence, entitiesxsentence):
    labelsxsentence = []
    for tokens, entities in zip(tokensxsentence, entitiesxsentence):
        offset = 0
        labels = []
        for token in tokens:
            # Recently found that (token.idx, token.idx + len(token)) is the span
            matches = find_match(offset, offset + len(token.text), entities)
            tag = select_tag(matches)
            labels.append(tag)
            offset += len(token.text_with_ws)
        labelsxsentence.append(labels)

    return labelsxsentence  # , "BILUOV"


def find_match(start, end, entities):
    def match(other):
        return other[0] <= start and end <= other[1]

    matches = []
    for spans in entities:

        # UNIT
        if len(spans) == 1:
            if match(spans[0]):
                matches.append((spans[0], "U"))
            continue

        # BEGIN
        begin, *tail = spans
        if match(begin):
            matches.append((begin, "B"))
            continue

        # LAST
        *body, last = tail
        if match(last):
            matches.append((last, "L"))
            continue

        # INNER
        for inner in body:
            if match(inner):
                matches.append((inner, "I"))
                break

    return matches


def select_tag(matches):
    if not matches:
        return "O"
    if len(matches) == 1:
        return matches[0][1]
    tags = [tag for _, tag in matches]
    return "U" if ("U" in tags and not "B" in tags and not "L" in tags) else "V"


def make_sentence(doc, bilouv, labels) -> Sentence:
    sentence = Sentence(doc.text)

    logger.debug(f"[make_sentence]: doc.text={doc.text}")
    logger.debug(f"[make_sentence]: bilouv={bilouv}")

    labels = set(l[2:] for l in labels if l != "O")

    for label in labels:
        specific_bilouv = []

        for tag in bilouv:
            if tag.endswith(label):
                tag = tag[0]
                specific_bilouv.append(tag[0])
            else:
                specific_bilouv.append("O")

        logger.debug(
            f"[make_sentence]: label={label} specific_bilouv={specific_bilouv}"
        )

        spans = from_biluov(specific_bilouv, doc, spans=True)
        sentence.keyphrases.extend(
            Keyphrase(sentence, label, i, sp) for i, sp in enumerate(spans)
        )

    return sentence


def from_biluov(biluov, sentence, *, spans=False, drop_remaining=[]):
    """
    >>> from_biluov(list('BBULL'), 'A B C D E'.split())
    [['C'], ['B', 'D'], ['A', 'E']]
    """

    entities = [x for x in discontinuous_match(biluov, sentence)]

    for i, (tag, word) in enumerate(zip(biluov, sentence)):
        if tag == "U":
            entities.append([word])
            biluov[i] = "O"
        elif tag == "V":
            biluov[i] = "I"

    # only BILO is left!!!
    changed = True
    while changed:
        changed = False
        one_shot = enumerate(zip(biluov, sentence))
        try:
            i, (tag, word) = next(one_shot)
            while True:
                if tag != "B":
                    i, (tag, word) = next(one_shot)
                    continue

                on_build = [(word, i)]

                i, (tag, word) = next(one_shot)
                while tag in ("O", "I"):
                    if tag == "I":
                        on_build.append(word)
                    i, (tag, word) = next(one_shot)

                if tag == "L":
                    entities.append([x for x, _ in on_build] + [word])
                    for _, j in on_build:
                        biluov[j] = "O"
                    biluov[i] = "O"
                    on_build.clear()
                    changed = True
        except StopIteration:
            pass

    for i, (tag, word) in enumerate(zip(biluov, sentence)):
        if tag != "O" and tag not in drop_remaining:
            entities.append([word])

    return (
        entities
        if not spans
        else [[(t.idx, t.idx + len(t)) for t in tokens] for tokens in entities]
    )


def discontinuous_match(biluov, sentence):
    """
    >>> discontinuous_match(['B','V','L'],['la', 'enfermedad', 'renal'])
    [['la', 'enfermedad', 'renal'], ['enfermedad']]
    >>> discontinuous_match(['O','V','I','L','O','I','L'],['el','cancer','de','pulmon','y','de','mama'])
    [['cancer', 'de', 'pulmon'], ['cancer', 'de', 'mama']]
    >>> discontinuous_match(['B','O','B','V'],['tejidos','y','organos','humanos'])
    [['organos', 'humanos'], ['tejidos', 'humanos']]
    >>> discontinuous_match(['O','V','I','L','O','I','L','O','B','O','B','V'], ['el','cancer','de','pulmon','y','de','mama','y','tejidos','y','organos','humanos'])
    [['cancer', 'de', 'pulmon'], ['cancer', 'de', 'mama'], ['organos', 'humanos'], ['tejidos', 'humanos']]
    >>> discontinuous_match(list('BBULL'), 'A B C D E'.split())
    []
    """
    entities = []
    for i, tag in enumerate(biluov):
        if tag != "V":
            continue
        for entity_ids in _full_overlap(biluov, list(range(len(sentence))), i):
            entity = []
            for idx in entity_ids:
                entity.append(sentence[idx])
                biluov[idx] = "O"
            entities.append(entity)
    return entities


def _full_overlap(biluov, sentence, index, product=False):
    """
    INDEX TAG MUST BE 'V'
    >>> _full_overlap(['B','V','L'], list(range(3)), 1)
    [[0, 1, 2], [1]]
    >>> _full_overlap(['B','V','V','L'], list(range(4)), 1)
    [[0, 1, 2, 3], [1, 2]]
    >>> _full_overlap(['B','V','V','L'], list(range(4)), 2)
    [[0, 1, 2, 3], [1, 2]]
    >>> _full_overlap(['B','V','V','V','L'], list(range(5)), 1)
    [[0, 1, 2, 3, 4], [1, 2, 3]]
    >>> _full_overlap(['B','V','V','V','L'], list(range(5)), 2)
    [[0, 1, 2, 3, 4], [1, 2, 3]]
    >>> _full_overlap(['B','V','V','V','L'], list(range(5)), 3)
    [[0, 1, 2, 3, 4], [1, 2, 3]]
    >>> _full_overlap(['B','B','V','L','L'], list(range(5)), 2)
    [[1, 2, 3], [0, 2, 4]]
    >>> _full_overlap(['B','I','B','O','V','I','L','O','L'], list(range(9)), 4)
    [[2, 4, 5, 6], [0, 1, 4, 8]]
    >>> _full_overlap(['B','I','B','O','V','I','L','O','L'], list(range(9)), 4, True)
    [[2, 4, 5, 6], [2, 4, 8], [0, 1, 4, 5, 6], [0, 1, 4, 8]]
    >>> _full_overlap(['0','0','V','L'], list(range(4)), 2)
    [[2, 3], [2]]
    >>> _full_overlap(['V','L'], list(range(2)), 0)
    [[0, 1], [0]]
    >>> _full_overlap(['B','V','O','O'], list(range(4)), 1)
    [[0, 1], [1]]
    >>> _full_overlap(['B','V'], list(range(2)), 1)
    [[0, 1], [1]]
    >>> _full_overlap(['0','0','V','O','O'], list(range(5)), 2)
    []
    """

    left = _right_to_left_overlap(biluov[: index + 1], sentence[: index + 1])
    right = _left_to_right_overlap(biluov[index:], sentence[index:])

    full = []
    if product:
        for l in left:
            for r in right:
                new = l + r[1:] if len(l) > len(r) else l[:-1] + r
                full.append(new)
    else:
        for l, r in itt.zip_longest(left, right, fillvalue=[]):
            new = l + r[1:] if len(l) > len(r) else l[:-1] + r
            full.append(new)
    return full


def _left_to_right_overlap(biluov, sentence):
    """
    LEFTMOST TAG MUST BE 'V'
    >>> _left_to_right_overlap(['V', 'V', 'O', 'V', 'I', 'L', 'O', 'I', 'L'], range(9))
    [[0, 1, 3, 4, 5], [0, 1, 3, 7, 8]]
    >>> _left_to_right_overlap(['V', 'O', 'V', 'O'], range(4))
    []
    >>> _left_to_right_overlap(['V', 'O', 'V', 'O', 'L'], range(5))
    [[0, 2, 4], [0, 2]]
    >>> _left_to_right_overlap(['V', 'O', 'V', 'O', 'L', 'O', 'L'], range(8))
    [[0, 2, 4], [0, 2, 6]]
    >>> _left_to_right_overlap(['V', 'O', 'V', 'O', 'L', 'I', 'L', 'V', 'L'], range(9))
    [[0, 2, 4], [0, 2, 5, 6]]
    """
    return _build_overlap(biluov, sentence, "L")


def _right_to_left_overlap(biluov, sentence):
    """
    RIGHTMOST TAG MUST BE 'V'
    >>> _right_to_left_overlap(['B', 'I', 'O', 'B', 'I', 'V', 'O', 'V', 'V'], range(9))
    [[3, 4, 5, 7, 8], [0, 1, 5, 7, 8]]
    >>> _right_to_left_overlap(['O', 'V', 'O', 'V'], range(4))
    []
    >>> _right_to_left_overlap(['B', 'O', 'V', 'O', 'V'], range(5))
    [[0, 2, 4], [2, 4]]
    >>> _right_to_left_overlap(['B', 'O', 'B', 'O', 'V', 'O', 'V'], range(7))
    [[2, 4, 6], [0, 4, 6]]
    >>> _right_to_left_overlap(['B', 'V', 'B', 'I', 'B', 'O', 'V', 'O', 'V'], range(9))
    [[4, 6, 8], [2, 3, 6, 8]]
    """
    inverse = _build_overlap(reversed(biluov), reversed(sentence), "B")
    for x in inverse:
        x.reverse()
    return inverse


def _build_overlap(biluov, sentence, finisher):
    """
    LEFTMOST TAG MUST BE 'V'
    """

    one_shot = zip(biluov, sentence)
    tag, word = next(one_shot)

    prefix = []
    complete = []

    try:
        while tag in ("V", "O"):
            if tag == "V":
                prefix.append(word)
            tag, word = next(one_shot)

        on_build = []
        while tag in ("O", "I", "U", finisher):
            if tag == "I":
                on_build.append(word)
            elif tag == finisher:
                complete.append(prefix + on_build + [word])
                on_build.clear()
            elif tag == "U":
                complete.append([word])
            tag, word = next(one_shot)
    except StopIteration:
        pass

    if len(complete) == 1:
        complete.append(prefix)

    return complete
