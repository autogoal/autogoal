import json
import numpy as np
import os
from autogoal.datasets import datapath, download
import csv
import re
import enum

class TaskType(enum.Enum):
    TokenClassification="TokenClassification",
    SentenceClassification="SentenceClassification"
    TextClassification="TextClassification"
    
def load(
    mode=TaskType.TokenClassification,
    include_sentiment=True,
    include_emotions=True,
    verbose=False):
    """
    Loads full dataset from [IntentSEMO].
    """

    try:
        download("intentsemo")
    except:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    path = datapath("intentsemo")
    
    X_train_raw = []
    X_train = []
    y_train_raw = []
    y_train=[]
    
    X_test_raw = []
    X_test = []
    y_test_raw = []
    y_test=[]
    
    with open(path / "train.csv", "r") as fd:
        reader = csv.reader(fd)
        
        if (mode == TaskType.TokenClassification):
            _, _, X_train, y_train = load_tokens(reader, include_sentiment, include_emotions, verbose)
        
        if (mode == TaskType.SentenceClassification):
            _, _, X_train, y_train = load_sentences(reader,include_sentiment,include_emotions, True, verbose)
        
        if (mode == TaskType.TextClassification):
            _, _, X_train, y_train = load_texts(reader,include_sentiment,include_emotions, True, verbose)
    
    with open(path / "test.csv", "r") as fd:
        reader = csv.reader(fd)
        
        if (mode == TaskType.TokenClassification):
            _, _, X_test, y_test = load_tokens(reader, include_sentiment, include_emotions, verbose)
        
        if (mode == TaskType.SentenceClassification):
            _, _, X_test, y_test = load_sentences(reader,include_sentiment,include_emotions, True, verbose)
        
        if (mode == TaskType.TextClassification):
            _, _, X_test, y_test = load_texts(reader,include_sentiment,include_emotions, True, verbose)
    
    return X_train, y_train, X_test, y_test
        
def load_tokens(
    reader, 
    include_sentiment=True,
    include_emotions=True,
    verbose=False):
    X_raw= []
    X = []
    y_raw = []
    y = []
    
    ignore_count = 0
    for row in reader:
        if ignore_count < 1:
            ignore_count+=1
            continue
        
        start_index = 14
        end_index = 15
        text_index = 0
        
        sentiment_neu_intex = 2
        sentiment_pos_intex = 3
        sentiment_neg_intex = 4
        
        emotion_others_index = 6
        emotion_joy_index = 7
        emotion_surprise_index = 8
        emotion_fear_index = 9
        emotion_sadness_index = 10
        emotion_anger_index = 11
        emotion_disgust_index = 12
        
        label_index = 13
        
        entities = [{"start": int(row[start_index]), "end": int(row[end_index]), "label": row[label_index]}]
        
        text = row[text_index]
        
        sentiment = [float(row[sentiment_neu_intex]), 
                     float(row[sentiment_pos_intex]), 
                     float(row[sentiment_neg_intex])]
        
        emotions = [float(row[emotion_others_index]), 
                    float(row[emotion_joy_index]), 
                    float(row[emotion_surprise_index]), 
                    float(row[emotion_fear_index]), 
                    float(row[emotion_sadness_index]), 
                    float(row[emotion_anger_index]), 
                    float(row[emotion_disgust_index])]
        
        tokens, iobtags = span_to_iob(text, entities)
        X_raw.append(text)
        y_raw.append(entities)
        
        assert len(tokens) == len(iobtags)
        
        x = [tokens]
        if (include_sentiment):
            x.extend(sentiment)
        
        if (include_emotions):
            x.extend(emotions)
            
        X.append(tuple(x))
        y.append(iobtags)
        
    assert len(X) == len(y)
    assert len(X_raw) == len(y_raw)
    assert len(X_raw) == len(X)
    assert len(y_raw) == len(y)
    
    if (verbose):
        print(f"Loaded {len(X)} items.")
    
    for i in range(len(X)):
        assert len(X[i][0]) == len(y[i])
    
    return X_raw, y_raw, X, y
   
def load_sentences(
    reader, 
    include_sentiment=True,
    include_emotions=True,
    single_label = True, 
    verbose=False):
    X_raw= []
    X = []
    y_raw = []
    y = []
    
    ignore_count = 0
    for row in reader:
        if ignore_count < 1:
            ignore_count+=1
            continue
        
        text_index = 0
        
        sentiment_neu_intex = 1
        sentiment_pos_intex = 2
        sentiment_neg_intex = 3
        
        emotion_others_index = 4
        emotion_joy_index = 5
        emotion_surprise_index = 6
        emotion_fear_index = 7
        emotion_sadness_index = 8
        emotion_anger_index = 9
        emotion_disgust_index = 10
        
        label_index = 11
        
        text = row[text_index]
        
        sentiment = [float(row[sentiment_neu_intex]), 
                     float(row[sentiment_pos_intex]), 
                     float(row[sentiment_neg_intex])]
        
        emotions = [float(row[emotion_others_index]), 
                    float(row[emotion_joy_index]), 
                    float(row[emotion_surprise_index]), 
                    float(row[emotion_fear_index]), 
                    float(row[emotion_sadness_index]), 
                    float(row[emotion_anger_index]), 
                    float(row[emotion_disgust_index])]
        
        sentences, labels = [text], [row[label_index]] #span_to_sentence_class(text, entities) if single_label else span_to_sentence_multilabel_class(text, entities)
        
        for sentence in sentences:
            additional = []
            if (include_sentiment):
                additional.extend(sentiment)
            
            if (include_emotions):
                additional.extend(emotions)
                
            X.append((sentence, additional))
        y.extend(labels)
        
    assert X.count('') == 0
    assert len(X) == len(y)
    assert len(X_raw) == len(y_raw)
    
    if (verbose):
        print(f"Loaded {len(X)} items.")
    
    return X_raw, y_raw, X, y
  
def load_texts(
    reader, 
    include_sentiment=True,
    include_emotions=True,
    single_label = True, 
    verbose=False):
    X_raw= []
    X = []
    y_raw = []
    y = []
    
    ignore_count = 0
    for row in reader:
        if ignore_count < 1:
            ignore_count+=1
            continue
        
        start_index = 14
        end_index = 15
        text_index = 0
        
        sentiment_neu_intex = 2
        sentiment_pos_intex = 3
        sentiment_neg_intex = 4
        
        emotion_others_index = 6
        emotion_joy_index = 7
        emotion_surprise_index = 8
        emotion_fear_index = 9
        emotion_sadness_index = 10
        emotion_anger_index = 11
        emotion_disgust_index = 12
        
        label_index = 13
        label = row[label_index]
        
        text = row[text_index]
        
        sentiment = [float(row[sentiment_neu_intex]), 
                     float(row[sentiment_pos_intex]), 
                     float(row[sentiment_neg_intex])]
        
        emotions = [float(row[emotion_others_index]), 
                    float(row[emotion_joy_index]), 
                    float(row[emotion_surprise_index]), 
                    float(row[emotion_fear_index]), 
                    float(row[emotion_sadness_index]), 
                    float(row[emotion_anger_index]), 
                    float(row[emotion_disgust_index])]
        
        x = [text]
        if (include_sentiment):
            x.extend(sentiment)
        
        if (include_emotions):
            x.extend(emotions)
            
        X.append(tuple(x))
        y.append(label)
        
    assert X.count('') == 0
    assert len(X) == len(y)
    assert len(X_raw) == len(y_raw)
    
    if (verbose):
        print(f"Loaded {len(X)} items.")
    
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
    from nltk import sent_tokenize
    
    split_chars = [".", "!", "?", "\n"]
    
    # Sort the entities by start offset
    entities.sort(key=lambda x: x['start'])
    
    # Initialize the list of sentences and sentence labels
    sentences = []
    sentence_labels = []
    
    # Split the text into sentences
    # raw_sentences = re.split('|'.join(map(re.escape, split_chars)), text.replace('\n', ' '))
    # raw_sentences = [sentence.strip() for sentence in raw_sentences if sentence.strip()]  # Filter out empty sentences
    
    raw_sentences = sent_tokenize(text)
    
    # Initialize the current position in the text
    current_pos = 0
    
    for raw_sentence in raw_sentences:
        sentence_start = current_pos
        sentence_end = current_pos + len(raw_sentence)
        
        # Check if the sentence contains an entity
        contains_entity = False
        for entity in entities:
            if not (entity['end'] < sentence_start or entity['start'] > sentence_end):
                contains_entity = True
                label = entity['label']
                break
        
        # Add the sentence and its label to the lists
        sentences.append(raw_sentence)
        if contains_entity:
            sentence_labels.append(label)
        else:
            sentence_labels.append('O')
        
        # Update the current position
        current_pos = sentence_end + 1  # +1 to account for the split character
    
    return sentences, [label if label is not None else 'None' for label in sentence_labels]

def span_to_iob(text, entities):
    # Define the characters to split on
    split_chars = ["\n", " ", ",", ".", ";", ":", "!", "?", "(", ")"]
    
    # Sort the entities by start offset
    entities.sort(key=lambda x: x['start'])
    
    # Initialize the list of tokens and labels
    tokens = []
    labels = []
    
    # Initialize the current position in the text
    current_pos = 0
    
    for entity in entities:
        start = entity['start']
        end = entity['end']
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
        labels.extend(([f'{label}'] if len(entity_tokens) > 0 else []) + [f'{label}'] * (len(entity_tokens) - 1))
        
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

def macro_f1(y, predicted, *args, **kwargs):
    y_flat = [tag for sublist in y for tag in sublist]
    predicted_flat = [tag for sublist in predicted for tag in sublist]

    return macro_f1_plain(y_flat, predicted_flat)

def weighted_f1(y, predicted, *args, **kwargs):
    y_flat = [tag for sublist in y for tag in sublist]
    predicted_flat = [tag for sublist in predicted for tag in sublist]

    return weighted_f1_plain(y_flat, predicted_flat)


def macro_f1_plain(y, predicted, *args, **kwargs):
    """
    Macro-average F1 evaluation function
    """
    from sklearn.metrics import f1_score
    
    # Get the unique classes
    return f1_score(y, predicted, average='macro')

def weighted_f1_plain(y, predicted, *args, **kwargs):
    """
    Macro-average F1 evaluation function
    """
    from sklearn.metrics import f1_score
    
    # Get the unique classes
    return f1_score(y, predicted, average='weighted')

if __name__ == "__main__":
    load(TaskType.SentenceClassification)