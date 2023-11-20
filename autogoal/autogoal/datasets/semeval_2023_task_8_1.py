import json
import numpy as np
import os
from autogoal.datasets import datapath, download
import csv
import re
import enum

class TaskTypeSemeval(enum.Enum):
    TokenClassification="TokenClassification",
    SentenceClassification="SentenceClassification",
    SentenceMultilabelClassification="SentenceMultilabelClassification"
    
class SemevalDatasetSelection(enum.Enum):
    Original="Original",
    Actual="Actual",

def load(mode=TaskTypeSemeval.TokenClassification, data_option=SemevalDatasetSelection.Original, verbose=False):
    """
    Loads full dataset from [Semeval 2023 Task 8.1](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN).
    """

    try:
        download("semeval_2023_t8.1")
    except:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    path = datapath("semeval_2023_t8.1")
    
    X_train_raw = []
    X_train = []
    y_train_raw = []
    y_train=[]
    
    X_test_raw = []
    X_test = []
    y_test_raw = []
    y_test=[]
    
    if (data_option == SemevalDatasetSelection.Original):
        # load train
        with open(path / "st1_train.csv", "r") as fd:
            reader = csv.reader(fd)
            
            if (mode == TaskTypeSemeval.TokenClassification):
                X_train_raw, y_train_raw, X_train, y_train = load_tokens(reader, verbose)
            
            if (mode == TaskTypeSemeval.SentenceClassification):
                X_train_raw, y_train_raw, X_train, y_train = load_sentences(reader, True, verbose)
            
            if (mode == TaskTypeSemeval.SentenceMultilabelClassification):
                X_train_raw, y_train_raw, X_train, y_train = load_sentences(reader, False, verbose)
            
        # load test
        # with open(path / "st1_test.csv", "r") as fd:
        #     reader = csv.reader(fd)
            
        #     if (mode == TaskTypeSemeval.TokenClassification):
        #         X_test_raw, y_test_raw, X_test, y_test = load_tokens(reader)
            
        #     if (mode == TaskTypeSemeval.SentenceClassification):
        #         X_test_raw, y_test_raw, X_test, y_test = load_sentences(reader)
            
        #     if (mode == TaskTypeSemeval.SentenceMultilabelClassification):
        #         X_test_raw, y_test_raw, X_test, y_test = load_sentences(reader, False)
                
    
    else:
        
        # load train
        with open(path / "st1_actual.csv", "r") as fd:
            reader = csv.reader(fd)
            
            if (mode == TaskTypeSemeval.TokenClassification):
                X_train_raw, y_train_raw, X_train, y_train = load_tokens(reader, verbose)
            
            if (mode == TaskTypeSemeval.SentenceClassification):
                X_train_raw, y_train_raw, X_train, y_train = load_sentences(reader, True, verbose)
            
            if (mode == TaskTypeSemeval.SentenceMultilabelClassification):
                X_train_raw, y_train_raw, X_train, y_train = load_sentences(reader, False, verbose)
            
    return X_train, y_train, X_test, y_test
        
    
def load_tokens(reader, verbose=False):
    X_raw= []
    X = []
    y_raw = []
    y = []
    
    title_line = True
    invalid_rows = 0
    for row in reader:
        if title_line:
            title_line = False
            continue
        
        try:
            rawl = json.loads(row[2])
        except:
            if (verbose):
                print(f"Invalid row, annotation not recognized at line {reader.line_num}.")
            continue
            
        if (len(rawl) > 1 and verbose):
            print(f"Warning, multiple annotations detected in line {reader.line_num}!!")
            
        entities = rawl[0]['crowd-entity-annotation']['entities']
        
        text = row[3]
        
        if len(text) == 0: # Invalid row
            if (verbose):
                print(f"Invalid row, no text detected for line {reader.line_num}.")
            invalid_rows += 1
            continue
        
        if (text == '[deleted by user]\n[removed]'): # Invalid row
            if (verbose):
                print(f"Invalid row, referenced comment at line {reader.line_num} was deleted.")
            invalid_rows += 1
            continue
        
        tokens, iobtags = span_to_iob(text, entities)
        X_raw.append(text)
        y_raw.append(entities)
        
        assert len(tokens) == len(iobtags)
        
        X.append(tokens)
        y.append(iobtags)
        
    assert len(X) == len(y)
    assert len(X_raw) == len(y_raw)
    assert len(X_raw) == len(X)
    assert len(y_raw) == len(y)
    
    if (verbose):
        print(f"Loaded {len(X)} items. A total of {invalid_rows} rows were invalid.")
    
    return X_raw, y_raw, X, y
   
def load_sentences(reader, single_label = True, verbose=False):
    X_raw= []
    X = []
    y_raw = []
    y = []
    
    title_line = True
    invalid_rows = 0
    for row in reader:
        if title_line:
            title_line = False
            continue
        
        try:
            rawl = json.loads(row[2])
        except:
            if (verbose):
                print(f"Invalid row, annotation not recognized at line {reader.line_num}.")
            continue
            
        if (len(rawl) > 1 and verbose):
            print(f"Warning, multiple annotations detected in line {reader.line_num}!!")
            
        entities = rawl[0]['crowd-entity-annotation']['entities']
        
        text = row[3]
        
        if len(text) == 0: # Invalid row
            if (verbose):
                print(f"Invalid row, no text detected for line {reader.line_num}.")
            invalid_rows += 1
            continue
        
        if (text == '[deleted by user]\n[removed]'): # Invalid row
            if (verbose):
                print(f"Invalid row, referenced comment at line {reader.line_num} was deleted.")
            invalid_rows += 1
            continue
        
        sentences, labels = span_to_sentence_class(text, entities) if single_label else span_to_sentence_multilabel_class(text, entities)
        X.extend(sentences)
        y.extend(labels)
        
    assert len(X) == len(y)
    assert len(X_raw) == len(y_raw)
    
    if (verbose):
        print(f"Loaded {len(X)} items. A total of {invalid_rows} rows were invalid.")
    
    return X_raw, y_raw, X, y
        
def span_to_sentence_multilabel_class(text, entities):
    # Define the characters to split on to get sentences
    split_chars = [".", "!", "?", "\n"]
    
    # Split the text into sentences
    sentences = [sentence.strip() for sentence in re.split('|'.join(map(re.escape, split_chars)), text) if sentence.strip()]
    
    # Initialize the list of sentence labels
    sentence_labels = [[] for _ in sentences]
    
    # Assign labels to the sentences based on the entities they contain
    for entity in entities:
        start = entity['startOffset']
        end = entity['endOffset']
        label = entity['label']
        
        for i, sentence in enumerate(sentences):
            sentence_start = text.index(sentence)
            sentence_end = sentence_start + len(sentence)
            
            if (start >= sentence_start and start <= sentence_end) or (end >= sentence_start and end <= sentence_end):
                sentence_labels[i].append(label)
    
    # Convert the list of labels for each sentence to a set to remove duplicates
    sentence_labels = [list(set(labels)) for labels in sentence_labels]
    
    return sentences, sentence_labels

def span_to_sentence_class(text, entities):
    # Define the characters to split on to get sentences
    split_chars = [".", "!", "?", "\n"]
    
    # Sort the entities by start offset
    entities.sort(key=lambda x: x['startOffset'])
    
    # Initialize the list of sentences and sentence labels
    sentences = []
    sentence_labels = []
    
    # Initialize the current position in the text
    current_pos = 0
    
    for entity in entities:
        start = entity['startOffset']
        end = entity['endOffset']
        label = entity['label']
        
        # Add the text before the entity (if any) as separate sentences
        if start > current_pos:
            sentences_before = re.split('|'.join(map(re.escape, split_chars)), text[current_pos:start].strip().replace('\n', ' '))
            sentences.extend(sentence.strip() for sentence in sentences_before if sentence)  # Filter out empty sentences
            sentence_labels.extend(['O' for sentence in sentences_before if sentence])
        
        # Extend the end of the entity to the next sentence boundary
        for sent_boundary in split_chars:
            end = text.find(sent_boundary, end)
            if (end != -1):
                break
            
        # If there's no sentence boundary, use the end of the text
        if end == -1:
            end = len(text)
        
        # Add the entity as a separate sentence
        sentences.append(text[start:end].replace('\n', ' ').strip())
        sentence_labels.append(label)
        
        # Update the current position
        current_pos = end
    
    # Add the text after the last entity (if any) as separate sentences
    if current_pos < len(text):
        sentences_after = re.split('|'.join(map(re.escape, split_chars)), text[current_pos:].strip().replace('\n', ' '))
        sentences.extend(sentence.strip() for sentence in sentences_after if sentence)  # Filter out empty sentences
        sentence_labels.extend(['O' for sentence in sentences_after if sentence])
    
    return sentences, [ label if label is not None else 'None' for label in sentence_labels]

def span_to_iob(text, entities):
    # Define the characters to split on
    split_chars = ["\n", " ", ",", ".", ";", ":", "!", "?", "(", ")"]
    
    # Sort the entities by start offset
    entities.sort(key=lambda x: x['startOffset'])
    
    # Initialize the list of tokens and labels
    tokens = []
    labels = []
    
    # Initialize the current position in the text
    current_pos = 0
    
    for entity in entities:
        start = entity['startOffset']
        end = entity['endOffset']
        label = entity['label']
        
        # Tokenize the text before the entity (if any) and add it to the tokens and labels
        if start > current_pos:
            tokens_before = re.split('|'.join(map(re.escape, split_chars)), text[current_pos:start].strip())
            tokens_before = [token.strip() for token in tokens_before if token.strip()]
            tokens.extend(tokens_before)  # Filter out empty tokens
            labels.extend(['O'] * len(tokens_before))
        
        # Add the entity to the tokens and labels
        entity_tokens = text[start:end].split()
        tokens.extend(entity_tokens)
        labels.extend(([f'B-{label}'] if len(entity_tokens) > 0 else []) + [f'I-{label}'] * (len(entity_tokens) - 1))
        
        # Update the current position
        current_pos = end
    
    # Tokenize the text after the last entity (if any) and add it to the tokens and labels
    if current_pos < len(text):
        tokens_after = re.split('|'.join(map(re.escape, split_chars)), text[current_pos:].strip())
        tokens_after = [token.strip() for token in tokens_after if token.strip()]
        tokens.extend(tokens_after)  # Filter out empty tokens
        labels.extend(['O'] * len(tokens_after))
    
    return tokens, labels

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



def get_qvals_plain(y, predicted):
    tp = 0
    fp = 0
    fn = 0
    total_sentences = 0
    for i in range(len(y)):
        tag = y[i]
        predicted_tag = predicted[i]
        if tag != "O":
            if tag == predicted_tag:
                tp += 1
            else:
                fn += 1
        elif tag != predicted_tag:
            fp += 1
            
        total_sentences += 1

    return tp, fp, fn, total_sentences

def leak_plain(y, predicted):
    """
    leak evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    tp, fp, fn, total_sentences = get_qvals_plain(y, predicted)
    try:
        return float(fn / total_sentences)
    except ZeroDivisionError:
        return 0.0

def precision_plain(y, predicted):
    """
    precision evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    tp, fp, fn, total_sentences = get_qvals_plain(y, predicted)
    try:
        return tp / float(tp + fp)
    except ZeroDivisionError:
        return 0.0

def recall_plain(y, predicted):
    """
    recall evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    tp, fp, fn, total_sentences = get_qvals_plain(y, predicted)
    try:
        return tp / float(tp + fn)
    except ZeroDivisionError:
        return 0.0

def F1_beta_plain(y, predicted, beta=1):
    """
    F1 evaluation function from [MEDDOCAN iberleaf 2018](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN)
    """
    p = precision_plain(predicted, y)
    r = precision_plain(predicted, y)
    try:
        return (1 + beta**2) * ((p * r) / (p + r))
    except ZeroDivisionError:
        return 0.0
        pass

def basic_fn_plain(y, predicted):
    correct = 0
    total = 0
    for i in range(len(y)):
        for j in range(len(y[i])):
            total += 1

            _, tag = y[i][j]
            _, predicted_tag = predicted[i][j]
            correct += 1 if tag == predicted_tag else 0

    return correct / total

# load(mode=TaskTypeSemeval.TokenClassification, data_option=SemevalDatasetSelection.Actual)
