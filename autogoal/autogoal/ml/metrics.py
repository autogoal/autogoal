import inspect
from textwrap import wrap
import numpy as np
import statistics
from autogoal.ml.utils import LabelEncoder, check_number_of_labels
from functools import wraps
from deprecated import deprecated

METRICS = []


def register_metric(func):
    METRICS.append(func)
    return func


def find_metric(*types):
    for metric_func in METRICS:
        signature = inspect.signature(metric_func)

        if len(types) != len(signature.parameters):
            break

        for type_if, type_an in zip(types, signature.parameters):
            if not conforms(type_an.annotation, type_if):
                break

        return metric_func

    raise ValueError("No metric found for types: %r" % types)


def supervised_fitness_fn_moo(objectives):
    """
    Returns a fitness function for multi-objective optimization problems.

    Args:
    - objectives: a list of objective functions to optimize

    Returns:
    - fitness_fn: a function that takes a pipeline, a dataset (X, y), and optional arguments,
                  and returns a tuple of scores for each objective function
    """

    def fitness_fn(
        pipeline,
        X,
        y,
        *args,
        validation_split=0.3,
        cross_validation_steps=3,
        cross_validation="median",
        **kwargs
    ):
        """
        Performs cross-validation to evaluate the performance of a pipeline on a dataset.

        Args:
        - pipeline: the pipeline to evaluate
        - X: the input data
        - y: the output data
        - validation_split: the proportion of data to use for validation
        - cross_validation_steps: the number of times to perform cross-validation
        - cross_validation: the function to use to aggregate the cross-validation scores (either 'mean' or 'median')
        - kwargs: additional arguments to pass to the pipeline

        Returns:
        - r_scores: a tuple of scores for each objective function, aggregated over the cross-validation steps
        """

        scores = []
        for _ in range(cross_validation_steps):
            # Split the data into training and validation sets
            len_x = len(X) if isinstance(X, list) else X.shape[0]
            indices = np.arange(0, len_x)
            np.random.shuffle(indices)
            split_index = int(validation_split * len(indices))
            train_indices = indices[:-split_index]
            test_indices = indices[-split_index:]

            # Split the data into training and validation sets
            if isinstance(X, list):
                X_train, y_train, X_test, y_test = (
                    [X[i] for i in train_indices],
                    y[train_indices],
                    [X[i] for i in test_indices],
                    y[test_indices],
                )
            else:
                X_train, y_train, X_test, y_test = (
                    X[train_indices],
                    y[train_indices],
                    X[test_indices],
                    y[test_indices],
                )

            # Train the pipeline on the training set
            pipeline.send("train")
            pipeline.run(X_train, y_train, **kwargs)

            # Evaluate the pipeline on the validation set
            pipeline.send("eval")
            y_pred = pipeline.run(X_test, None, **kwargs)

            # Calculate the scores for each objective function
            scores.append([objective(y_test, y_pred) for objective in objectives])

        # Aggregate the scores over the cross-validation steps
        scores_per_objective = list(zip(*scores))
        r_scores = tuple(
            [
                getattr(statistics, cross_validation)(score)
                for score in scores_per_objective
            ]
        )
        return r_scores

    return fitness_fn


def unsupervised_fitness_fn_moo(objectives):
    """
    Returns a fitness function for unsupervised multi-objective optimization.

    Parameters:
    -----------
    objectives : list
        A list of objective functions to evaluate the performance of the unsupervised model.

    Returns:
    --------
    fitness_fn : function
        A fitness function that takes a pipeline and data X as inputs, runs the pipeline on X,
        and returns the tuple of objective scores.

    Example:
    --------
    >>> from sklearn.cluster import KMeans
    >>> pipeline = KMeans(n_clusters=2)
    >>> objectives = [silhouette_score, calinski_harabasz_score]
    >>> fitness_function = unsupervised_fitness_fn_moo(objectives)
    >>> scores = fitness_function(pipeline, X)
    """

    def fitness_fn(pipeline, X, *args, **kwargs):
        """
        Evaluates the performance of an unsupervised model using the given objectives.

        Parameters:
        -----------
        pipeline : object
            An unsupervised learning model that implements the fit and predict methods.

        X : array-like, shape (n_samples, n_features)
            The input data to train the model.

        Returns:
        --------
        tuple
            A tuple of objective scores.

        """
        pipeline.send("train")
        pipeline.run(X)
        pipeline.send("eval")
        y_pred = pipeline.run(X)
        return tuple([objective(X, y_pred) for objective in objectives])

    return fitness_fn


@deprecated(
    reason="This function is deprecated, please use the supervised_fitness_fn_moo() instead."
)
def supervised_fitness_fn(score_metric_fn):
    @wraps(score_metric_fn)
    def fitness_fn(
        pipeline,
        X,
        y,
        *args,
        validation_split=0.3,
        cross_validation_steps=3,
        cross_validation="median",
        **kwargs
    ):
        scores = []
        for _ in range(cross_validation_steps):
            len_x = len(X) if isinstance(X, list) else X.shape[0]
            indices = np.arange(0, len_x)
            np.random.shuffle(indices)
            split_index = int(validation_split * len(indices))
            train_indices = indices[:-split_index]
            test_indices = indices[-split_index:]

            if isinstance(X, list):
                X_train, y_train, X_test, y_test = (
                    [X[i] for i in train_indices],
                    y[train_indices],
                    [X[i] for i in test_indices],
                    y[test_indices],
                )
            else:
                X_train, y_train, X_test, y_test = (
                    X[train_indices],
                    y[train_indices],
                    X[test_indices],
                    y[test_indices],
                )

            pipeline.send("train")
            pipeline.run(X_train, y_train)
            pipeline.send("eval")
            y_pred = pipeline.run(X_test, None)
            scores.append(score_metric_fn(y_test, y_pred))
        return getattr(statistics, cross_validation)(scores)

    return fitness_fn


@deprecated(
    reason="This function is deprecated, please use the unsupervised_fitness_fn_moo() instead."
)
def unsupervised_fitness_fn(score_metric_fn):
    @wraps(score_metric_fn)
    def fitness_fn(pipeline, X, *args, **kwargs):
        scores = []
        pipeline.send("train")
        pipeline.run(X)
        pipeline.send("eval")
        y_pred = pipeline.run(X)
        return score_metric_fn(X, y_pred)

    return fitness_fn


def accuracy(ytrue, ypred) -> float:
    return np.mean([1 if yt == yp else 0 for yt, yp in zip(ytrue, ypred)])


def calinski_harabasz_score(X, labels):
    """Compute the Calinski and Harabasz score.

    It is also known as the Variance Ratio Criterion.

    The score is defined as ratio between the within-cluster dispersion and
    the between-cluster dispersion.

    Read more in the :ref:`User Guide <calinski_harabasz_index>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.

    References
    ----------
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_
    """
    # X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    check_number_of_labels(n_labels, n_samples)

    extra_disp, intra_disp = 0.0, 0.0
    mean = np.mean(X, axis=0)
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)

    return (
        1.0
        if intra_disp == 0.0
        else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
    )


def silhouette_score(X, labels):
    """
    Calculates the silhouette score for a given set of data and labels.

    Parameters:
    - X: a numpy array containing the data to be clustered
    - labels: a numpy array containing the labels assigned to each data point by a clustering algorithm

    Returns:
    - the silhouette score for the given clustering
    """

    n = len(X)
    s = np.zeros(n)

    # Calculate the average distance between each point and all other points in its cluster
    a = np.zeros(n)
    for i in range(n):
        cluster_indices = np.where(labels == labels[i])[0]
        a[i] = np.mean(np.linalg.norm(X[i] - X[cluster_indices], axis=1))

    # Calculate the average distance between each point and all points in the nearest neighboring cluster
    b = np.zeros(n)
    for i in range(n):
        other_cluster_indices = np.where(labels != labels[i])[0]
        min_distances = np.min(
            np.linalg.norm(X[i] - X[other_cluster_indices], axis=1)
        )
        b[i] = np.mean(min_distances)

    # Calculate the silhouette score for each point
    for i in range(n):
        s[i] = (b[i] - a[i]) / max(a[i], b[i])

    # Calculate the mean silhouette score for the entire clustering
    return np.mean(s)
