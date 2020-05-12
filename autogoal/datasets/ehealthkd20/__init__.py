from autogoal.datasets import download, datapath

from autogoal.datasets.ehealthkd20._utils import Collection


def load_training_entities():
    # download("ehealthkd20")

    training_path = datapath("ehealthkd20") / "training"

    collection = Collection().load_dir(training_path)
    
