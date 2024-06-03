import inspect
from textwrap import wrap
import time
from joblib import wrap_non_picklable_objects
import numpy as np
import statistics
from autogoal.ml.utils import LabelEncoder, check_number_of_labels, stratified_split_indices
from autogoal.kb import Pipeline
from functools import wraps
from deprecated import deprecated

RESOURCE_CONTROL_AVAILABLE = False

try:
    import resource
    RESOURCE_CONTROL_AVAILABLE = True
except ImportError:
    RESOURCE_CONTROL_AVAILABLE = False

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


def supervised_fitness_fn_moo(objectives, target_observations=None):
    """
    Returns a fitness function for multi-objective optimization problems.

    Args:
    - objectives: a list of objective functions to optimize
    - observations: a list of metrics for additional observations on the model.

    Returns:
    - fitness_fn: a function that takes a pipeline, a dataset (X, y), and optional arguments,
                  and returns a tuple of scores for each objective function and observations
    """

    def fitness_fn(
        pipeline: Pipeline,
        X,
        y,
        *args,
        validation_split=0.3,
        cross_validation_steps=3,
        cross_validation="median",
        stratified=True,
        pipeline_generator=None,
        target_observations=target_observations,
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
        - stratified: whether to use stratified k-fold cross-validation
        - kwargs: additional arguments to pass to the pipeline

        Returns:
        - r_scores: a tuple of scores for each objective function, aggregated over the cross-validation steps
        - observations: a tuple of scores for each observation function, aggregated over the cross-validation steps
        """
        original_pipeline = pipeline
        eval_pipeline = pipeline
        scores = []
        
        if target_observations is None:
            target_observations = []
            
        observations = {
            'time': {
                "train": [],
                "valid": []
            }
        }
        for _ in range(cross_validation_steps):
            
            X_instances = [X]
            if isinstance(X, tuple):
                X_instances = list(X)
                
            train_indices = []
            val_indices = []
            
            if stratified:
                train_indices, val_indices = stratified_split_indices(y, validation_split)
            else:
                # Split the data into training and validation sets
                len_x = len(X_instances[0]) if isinstance(X_instances[0], list) else X_instances[0].shape[0]
                indices = np.arange(0, len_x)
                np.random.shuffle(indices)
                split_index = int(validation_split * len(indices))
                train_indices = indices[:-split_index]
                val_indices = indices[-split_index:]
                
            X_train_instances = []
            X_val_instances = []
            for Xi in X_instances:
                # Split the data into training and validation sets
                X_train = []
                X_test = []
                
                if isinstance(Xi, list):
                    X_train, X_test = (
                        [Xi[i] for i in train_indices],
                        [Xi[i] for i in val_indices],
                    )
                else:
                    X_train, X_test = (
                        Xi[train_indices],
                        Xi[val_indices],
                    )
                    
                X_train_instances.append(X_train)
                X_val_instances.append(X_test)
                
            y_train = y[train_indices]
            y_test = y[val_indices]

            # if able, recreate the pipeline so it wont store much memory
            # This is additional security against memory leaks
            if (pipeline_generator is not None):
                original_pipeline.sampler_.replay()
                eval_pipeline = pipeline_generator(original_pipeline.sampler_)
            
            # Train the pipeline on the training set
            train_start_time = time.time()
            eval_pipeline.send("train")
            eval_pipeline.run(*X_train_instances, y_train, **kwargs)
            train_end_time = time.time()

            # Evaluate the pipeline on the validation set
            valid_start_time = time.time()
            eval_pipeline.send("eval")
            y_pred = eval_pipeline.run(*X_val_instances, None, **kwargs)
            valid_end_time = time.time()
            
            observations['time']['train'].append(train_end_time - train_start_time)
            observations['time']['valid'].append(valid_end_time - valid_start_time)

            # Calculate the scores for each objective function. We additionally pass 
            # evaluation_time always as users might want to select it for optimization
            # TODO: If we change how the pipeline training works this should be also targetted for improvement.
            scores.append([objective(y_test, y_pred, evaluation_time=valid_end_time - train_start_time) for objective in objectives])
            
            for (label, obs_func) in target_observations:
                if not label in observations:
                    observations[label] = []
                observations[label].append(obs_func(y_test, y_pred))
        
        # Aggregate the scores over the cross-validation steps
        scores_per_objective = list(zip(*scores))
        r_scores = tuple(
            [
                getattr(statistics, cross_validation)(score)
                for score in scores_per_objective
            ]
        )
        
        # Aggregate all observations
        for label, obs in observations.items():
            if label == 'time':
                continue
            observations[label] = getattr(statistics, cross_validation)(obs)
        
        return r_scores, observations

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


def accuracy(y, predictions, *args, **kwargs) -> float:
    zipped = zip(y, predictions)
    return np.mean([1 if yt == yp else 0 for yt, yp in zipped])

def peak_ram_usage(*args, **kwargs) -> float:
    if not RESOURCE_CONTROL_AVAILABLE:
        raise Exception("Peak Ram Consumed metric is not available on this system. Try installing the 'resource' package with 'pip install resource'.")
    
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss

def evaluation_time(*args, evaluation_time=None, **kwargs) -> float:
    return evaluation_time

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
        min_distances = np.min(np.linalg.norm(X[i] - X[other_cluster_indices], axis=1))
        b[i] = np.mean(min_distances)

    # Calculate the silhouette score for each point
    for i in range(n):
        s[i] = (b[i] - a[i]) / max(a[i], b[i])

    # Calculate the mean silhouette score for the entire clustering
    return np.mean(s)
