from autogoal.datasets import download, datapath

from autogoal.datasets.ehealthkd20._utils import Collection


def load_training() -> Collection:
    # download("ehealthkd20")

    training_path = datapath("ehealthkd20") / "training"

    return Collection().load_dir(training_path)
