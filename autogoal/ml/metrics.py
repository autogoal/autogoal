import inspect
import numpy as np
import statistics
import time
from autogoal.utils import PreemptiveStopException
from autogoal.ml.utils import LabelEncoder, check_number_of_labels

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


def supervised_fitness_fn(score_metric_fn):
    def fitness_fn(
        pipeline,
        X,
        y,
        *args,
        validation_split=0.3,
        cross_validation_steps=3,
        cross_validation="median",
        timeout=None,
        **kwargs,
    ):
        scores = []
        init_time = time.time()
        max_step_time = timeout / cross_validation_steps
        for step in range(cross_validation_steps):
            mean_step_time = (time.time() - init_time) / max(step, 1)
            if step != 0 and mean_step_time > max_step_time:
                raise PreemptiveStopException(
                    f"Preemptive stop: mean validation step time of {mean_step_time} is higher than expected {max_step_time}"
                )
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


def unsupervised_fitness_fn(score_metric_fn):
    def fitness_fn(pipeline, X, *args, **kwargs):
        scores = []
        pipeline.send("train")
        pipeline.run(X)
        pipeline.send("eval")
        y_pred = pipeline.run(X)
        return score_metric_fn(X, y_pred)

    return fitness_fn


@supervised_fitness_fn
def accuracy(ytrue, ypred) -> float:
    return np.mean([1 if yt == yp else 0 for yt, yp in zip(ytrue, ypred)])


@unsupervised_fitness_fn
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
