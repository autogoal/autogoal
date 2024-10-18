import enum
import random
import csv
from typing import Tuple, List, Optional, Callable

from autogoal.datasets import download, datapath
from sklearn.preprocessing import LabelEncoder

class TaskType(enum.Enum):
    TokenClassification = "TokenClassification"
    TextClassification = "TextClassification"

TASK_DATAPATH = {
    TaskType.TokenClassification: "intentes/intentes_token_classification",
    TaskType.TextClassification: "intentes/intentes_text_classification",
}

def load(
    *args,
    task: TaskType = TaskType.TextClassification,
    encoding: Optional[str] = None,
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
                texts.append(row[1])
                labels.append(row[2])
        return texts, labels
    
    # Read training and testing data
    X_train, y_train = read_csv(str(path / "train.csv"))
    X_test, y_test = read_csv(str(path / "test.csv"))
    
    # Encode labels if encoding is specified
    if encoding:
        if encoding.lower() == 'ordinal':
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train).tolist()
            y_test = label_encoder.transform(y_test).tolist()
            
            print("Classes:", label_encoder.classes_)
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
