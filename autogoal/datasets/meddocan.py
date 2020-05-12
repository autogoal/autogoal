import numpy as np
import os
from autogoal.datasets import datapath, download


def load(max_examples=None):
    """
    Loads train and test datasets from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN).

    ##### Examples

    ```python
    >>> X_train, y_train, X_valid, y_valid = load()
    >>> len(X_train), len(X_valid)
    (25622, 8432)
    >>> len(y_train), len(y_valid)
    (25622, 8432)

    ```
    """

    try:
        download("meddocan_2018")
    except:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    path = (
        str(datapath(os.path.dirname(os.path.abspath(__file__))))
        + "/data/meddocan_2018"
    )
    train_path = path + "/train/brat"
    dev_path = path + "/dev/brat"
    test_path = path + "/test/brat"

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    total = 0
    success = 0
    failed = 0

    for file in os.scandir(train_path):
        if file.name.split(".")[1] == "ann":
            text, phi = parse_text_and_tags(file.path)
            brat_corpora, text, ibo_corpora = get_tagged_tokens(text, phi)
            if compare_tags(brat_corpora, phi):
                X_train.extend(text)
                y_train.extend(ibo_corpora)

    for file in os.scandir(dev_path):
        if file.name.split(".")[1] == "ann":
            text, phi = parse_text_and_tags(file.path)
            brat_corpora, text, ibo_corpora = get_tagged_tokens(text, phi)
            if compare_tags(brat_corpora, phi):
                X_train.extend(text)
                y_train.extend(ibo_corpora)

    for file in os.scandir(test_path):
        if file.name.split(".")[1] == "ann":
            text, phi = parse_text_and_tags(file.path)
            brat_corpora, text, ibo_corpora = get_tagged_tokens(text, phi)
            if compare_tags(brat_corpora, phi):
                X_test.extend(text)
                y_test.extend(ibo_corpora)

    if max_examples is not None:
        X_train = X_train[:max_examples]
        X_test = X_test[:max_examples]
        y_train = y_train[:max_examples]
        y_test = y_test[:max_examples]

    return X_train, y_train, X_test, y_test


def parse_text_and_tags(file_name=None):
    """
    Given a file representing an annotated text in Brat format
    returns the `text` and `tags` annotated.
    """
    text = ""
    phi = []

    if file_name is not None:
        text = open(os.path.splitext(file_name)[0] + ".txt", "r").read()

        for row in open(file_name, "r"):
            line = row.strip()
            if line.startswith("T"):  # Lines is a Brat TAG
                try:
                    label = line.split("\t")[1].split()
                    tag = label[0]
                    start = int(label[1])
                    end = int(label[2])
                    value = text[start:end]
                    phi.append((tag, start, end, value))
                except IndexError:
                    print(
                        "ERROR! Index error while splitting sentence '"
                        + line
                        + "' in document '"
                        + file_name
                        + "'!"
                    )
            else:  # Line is a Brat comment
                print("\tSkipping line (comment):\t" + line)
    return (text, phi)


def get_tagged_tokens(text, tags):
    """
    convert a given text and annotations in brat format to IOB tag format

    Parameters:
    - text: raw text
    - tags: tags annotated on `text` with brat format

    output:
    tuple of identified tags in brat format from text and list of tokens tagged in IOB format
    """
    tags.sort(key=lambda x: x[1])
    offset = 0
    tagged_tokens = []

    current_tag = ""
    current_tag_end = 0
    current_tag_init = 0
    processing_token = False

    token = ""
    tag = ""

    itag = 0
    next_tag_init = tags[itag][1]

    sentences = [[]]

    for char in text:
        if processing_token and current_tag_end == offset:
            tagged_tokens.append((current_tag, current_tag_init, offset, token))

            tokens = token.split()
            if len(tokens) > 1:
                sentences[-1].append((tokens[0], tag))
                for tok in tokens[1:]:
                    sentences[-1].append((tok, "I-" + current_tag))
            else:
                sentences[-1].append((token, tag))

            token = ""
            current_tag = ""
            processing_token = False

        if not processing_token and char in [
            "\n",
            " ",
            ",",
            ".",
            ";",
            ":",
            "!",
            "?",
            "(",
            ")",
        ]:
            if token:
                sentences[-1].append((token, tag))

            if char in ["\n", ".", "!", " ?"] and len(sentences[-1]) > 1:
                sentences.append([])

            token = ""
            offset += 1
            continue

        if offset == next_tag_init:
            if token:
                if char in ["\n", " ", ",", ".", ";", ":", "!", "?", "(", ")"]:
                    sentences[-1].append((token, tag))
                else:
                    token += char
                    sentences[-1].append((token, tag))
                token = ""

            current_tag = tags[itag][0]
            current_tag_init = tags[itag][1]
            current_tag_end = tags[itag][2]
            processing_token = True

            itag += 1
            next_tag_init = tags[itag][1] if itag < len(tags) else -1

        if processing_token and current_tag:
            if not token:
                tag = "B-" + current_tag
        else:
            tag = "O"
        token += char
        offset += 1

    raw_sentences = [
        [word for word, _ in sentence] for sentence in sentences if sentence
    ]
    raw_tags = [[tag for _, tag in sentence] for sentence in sentences if sentence]
    return tagged_tokens, raw_sentences, raw_tags


def compare_tags(tag_list, other_tag_list):
    """
    compare two tags lists with the same tag format:

    (`tag_name`, `start_offset`, `end_offset`, `value`)
    """
    tags_amount = len(tag_list)

    if tags_amount != len(other_tag_list):
        print(
            "missmatch of amount of tags %d vs %d" % (tags_amount, len(other_tag_list))
        )
        return False

    tag_list.sort(key=lambda x: x[1])
    other_tag_list.sort(key=lambda x: x[1])
    for i in range(tags_amount):
        if len(tag_list[i]) != len(other_tag_list[i]):
            print("missmatch of tags format")
            return False

        for j in range(len(tag_list[i])):
            if tag_list[i][j] != other_tag_list[i][j]:
                print(
                    "missmatch of tags %s vs %s"
                    % (tag_list[i][j], other_tag_list[i][j])
                )
                return False

    return True


def get_qvals(y, predicted):
    tp = 0
    fp = 0
    fn = 0
    total_sentences = 0
    for i in range(len(y)):
        for j in range(len(y[i])):
            tag = y[i][j]
            predicted_tag = predicted[i][j]

            if tag != "O":
                if tag == predicted_tag:
                    tp += 1
                else:
                    fn += 1
            elif tag != predicted_tag:
                fp += 1
        total_sentences += 1

    return tp, fp, fn, total_sentences


def leak(y, predicted):
    """
    leak evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    tp, fp, fn, total_sentences = get_qvals(y, predicted)
    try:
        return float(fn / total_sentences)
    except ZeroDivisionError:
        return 0.0


def precision(y, predicted):
    """
    precision evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    tp, fp, fn, total_sentences = get_qvals(y, predicted)
    try:
        return tp / float(tp + fp)
    except ZeroDivisionError:
        return 0.0


def recall(y, predicted):
    """
    recall evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    tp, fp, fn, total_sentences = get_qvals(y, predicted)
    try:
        return tp / float(tp + fn)
    except ZeroDivisionError:
        return 0.0


def F1_beta(y, predicted, beta=1):
    """
    F1 evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    p = precision(predicted, y)
    r = recall(predicted, y)
    try:
        return (1 + beta ** 2) * ((p * r) / (p + r))
    except ZeroDivisionError:
        return 0.0
        pass


def basic_fn(y, predicted):
    correct = 0
    total = 0
    for i in range(len(y)):
        for j in range(len(y[i])):
            total += 1

            _, tag = y[i][j]
            _, predicted_tag = predicted[i][j]
            correct += 1 if tag == predicted_tag else 0

    return correct / total


def test_meddocan():
    X_train, X_valid, y_train, y_valid = load()

    assert len(X_train) == len(y_train)
    assert len(X_valid) == len(y_valid)

    assert all(isinstance(x, list) for x in X_train)
    assert all(isinstance(x, list) for x in X_valid)

    assert all(len(x) > 0 for x in X_train)
    assert all(len(x) > 0 for x in X_valid)

    for xi, yi in zip(X_train, y_train):
        assert len(xi) == len(yi)

    for xi, yi in zip(X_valid, y_valid):
        assert len(xi) == len(yi)
