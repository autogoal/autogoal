# AutoGOAL Example: basic usage of the AutoML class
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal.search import JsonLogger, ConsoleLogger
from autogoal.ml import AutoML
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster._unsupervised import silhouette_score as s_score
from autogoal.ml.metrics import unsupervised_fitness_fn_moo
from sklearn.decomposition import PCA
import argparse

#TODO: Fix this example. Unsupervised is not working as intended right now.

@unsupervised_fitness_fn_moo
def silhouette_score(X, labels):
    return s_score(X, labels)

parser = argparse.ArgumentParser()
parser.add_argument("--executions", type=int, default=1)
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--timeout", type=int, default=300)
parser.add_argument("--memory", type=int, default=10)
parser.add_argument("--popsize", type=int, default=100)
parser.add_argument("--selection", type=int, default=20)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--global-timeout", type=int, default=2 * 60)
parser.add_argument("--early-stop", type=int, default=20)
parser.add_argument("--token", default=None)
parser.add_argument("--channel", default=None)
args = parser.parse_args()

n_samples = 1000
for n_features in [2, 20, 100]:
    for centers in [3, 10, 20]:
        for cluster_std in [0.2, 0.5, 1.0]:
            for execution in range(args.executions):
                random_state = execution
                X, y = make_blobs(
                    n_samples=n_samples,
                    n_features=n_features,
                    centers=centers,
                    cluster_std=cluster_std,
                    random_state=random_state,
                )

                # # Instantiate AutoML, define input/output types and the score metric
                automl = AutoML(
                    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
                    output=VectorCategorical,
                    objectives=(silhouette_score),
                    search_iterations=args.iterations,
                    pop_size=args.popsize,
                    selection=args.selection,
                    evaluation_timeout=args.timeout,
                    memory_limit=args.memory * 1024**3,
                    early_stop=args.early_stop,
                    search_timeout=args.global_timeout,
                )

                loggers = [
                    ConsoleLogger()
                ]
                automl.fit(X, logger=loggers)
