import enum
import random
import csv
from typing import Tuple, List, Optional, Callable
import re

from autogoal.datasets import download, datapath
from sklearn.preprocessing import LabelEncoder

class TaskType(enum.Enum):
    TokenClassification = "TokenClassification"
    TextClassification = "TextClassification"

TASK_DATAPATH = {
    TaskType.TokenClassification: "intentes/intentes_token_classification",
    TaskType.TextClassification: "intentes/intentes_text_classification",
}

def de_emojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F92F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001F190-\U0001F1FF"
                                        u"\U0001F926-\U0001FA9F"
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u200d"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\ufe0f"
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def preprocess(value):
    new_value = de_emojify(value)
    new_value = re.sub(r'http\S+', '', new_value)
    return new_value


def load(
    *args,
    task: TaskType = TaskType.TextClassification,
    encoding: Optional[str] = "ordinal",
    **kwargs
) -> Tuple[List[str], List, List[str], List]:
    """
    Load the dataset for the specified task with optional class encoding.

    Parameters:
    - task (TaskType): The type of task to load.
    - encoding (Optional[str]): The encoding method for class labels. 
                                Supported: 'ordinal', 'onehot', etc.

    Returns:
    - Tuple containing training and testing data and labels.
    """
    try:
        download("intentes")
    except Exception as e:
        print(
            "Error loading data. This may be caused due to bad connection. "
            "Please delete badly downloaded data and retry."
        )
        raise e

    if task == TaskType.TokenClassification:
        return load_token_classification_dataset()
    elif task == TaskType.TextClassification:
        return load_text_classification_dataset(encoding=encoding)
    else:
        raise ValueError(f"Unsupported task type: {task}")

def load_text_classification_dataset(encoding: Optional[str] = None) -> Tuple[List[str], List, List[str], List]:
    """
    Load the text classification dataset with optional class encoding.

    Parameters:
    - encoding (Optional[str]): The encoding method for class labels.

    Returns:
    - Tuple containing training and testing data and encoded labels.
    """
    path = datapath(TASK_DATAPATH[TaskType.TextClassification])
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    # Helper function to read CSV files
    def read_csv(file_path: str) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []
        with open(file_path, "r", encoding='utf-8') as fd:
            reader = csv.reader(fd)
            header = next(reader, None)  # Skip header
            for row in reader:
                if len(row) < 3:
                    continue  # Skip malformed rows
                texts.append(preprocess(row[1]))
                labels.append(row[2])
        return texts, labels
    
    # Read training and testing data
    X_train, y_train = read_csv(str(path / "train.csv"))
    X_test, y_test = read_csv(str(path / "test.csv"))
    
    # Encode labels if encoding is specified
    if encoding:
        if encoding.lower() == 'ordinal':
            label_encoder = LabelEncoder()
            all_labels = y_train + y_test
            label_encoder.fit(all_labels)
            y_train = label_encoder.transform(y_train).tolist()
            y_test = label_encoder.transform(y_test).tolist()
            
            print("Encoded Classes and Mappings:")
            for class_label, class_index in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
                print(f"{class_label}: {class_index}")
        else:
            raise ValueError(f"Unsupported encoding type: {encoding}")
    
    return X_train, y_train, X_test, y_test

def load_token_classification_dataset():
    # Implementation for token classification
    pass

if __name__ == "__main__":
    # Example usage with ordinal encoding
    X_train, y_train, X_test, y_test = load(task=TaskType.TextClassification, encoding='ordinal')
    print("Training samples:", len(X_train))
    print("Training labels:", y_train[:5])
    print("Testing samples:", len(X_test))
    print("Testing labels:", y_test[:5])
