import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from autogoal.meta_learning.base import BaseFeatureExtractor

class TextClassificationFeatureExtractor(BaseFeatureExtractor):
    def extract_features(self, X_train, y_train, X_test, y_test) -> np.ndarray:
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Compute training set metrics
        n_instances_train = len(X_train)
        n_classes_train = len(np.unique(y_train))
        
        class_counts_train = np.array([np.sum(y_train == cls) for cls in np.unique(y_train)])
        class_probs_train = class_counts_train / n_instances_train
        class_entropy_train = -np.sum(class_probs_train * np.log2(class_probs_train + 1e-10))
        min_class_prob_train = np.min(class_probs_train)
        max_class_prob_train = np.max(class_probs_train)
        imbalance_ratio_train = min_class_prob_train / max_class_prob_train if max_class_prob_train > 0 else 0
        
        # Compute test set metrics
        n_instances_test = len(X_test)
        n_classes_test = len(np.unique(y_test))
        
        class_counts_test = np.array([np.sum(y_test == cls) for cls in np.unique(y_test)])
        class_probs_test = class_counts_test / n_instances_test
        class_entropy_test = -np.sum(class_probs_test * np.log2(class_probs_test + 1e-10))
        min_class_prob_test = np.min(class_probs_test)
        max_class_prob_test = np.max(class_probs_test)
        imbalance_ratio_test = min_class_prob_test / max_class_prob_test if max_class_prob_test > 0 else 0
        
        # Document lengths (in characters) for training and test sets
        doc_lengths_train = np.array([len(doc) for doc in X_train])
        avg_doc_length_train = np.mean(doc_lengths_train)
        std_doc_length_train = np.std(doc_lengths_train)
        coef_var_doc_length_train = std_doc_length_train / avg_doc_length_train if avg_doc_length_train != 0 else 0
        
        doc_lengths_test = np.array([len(doc) for doc in X_test])
        avg_doc_length_test = np.mean(doc_lengths_test)
        std_doc_length_test = np.std(doc_lengths_test)
        coef_var_doc_length_test = std_doc_length_test / avg_doc_length_test if avg_doc_length_test != 0 else 0
        
        # Landmarker: Decision Tree accuracy using cross-validation on training set
        sample_size = min(1000, n_instances_train)
        indices = np.random.choice(n_instances_train, size=sample_size, replace=False)
        X_sample_lengths = doc_lengths_train[indices].reshape(-1, 1)
        y_sample = y_train[indices]
        
        clf = DecisionTreeClassifier(max_depth=5)
        cv_scores = cross_val_score(clf, X_sample_lengths, y_sample, cv=5)
        decision_tree_accuracy_train = np.mean(cv_scores)
        
        # Combine all features into the feature vector
        feature_vector = np.array([
            
            # Training set metrics
            n_instances_train,
            n_classes_train,
            class_entropy_train,
            min_class_prob_train,
            max_class_prob_train,
            imbalance_ratio_train,
            avg_doc_length_train,
            std_doc_length_train,
            coef_var_doc_length_train,
            decision_tree_accuracy_train,
            
            # Test set metrics
            n_instances_test,
            n_classes_test,
            class_entropy_test,
            min_class_prob_test,
            max_class_prob_test,
            imbalance_ratio_test,
            avg_doc_length_test,
            std_doc_length_test,
            coef_var_doc_length_test,
            
            # Difference metrics (optional)
            n_instances_test / n_instances_train if n_instances_train > 0 else 0,
            class_entropy_test - class_entropy_train,
        ])
        
        return feature_vector