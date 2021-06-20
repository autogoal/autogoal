import pathlib
import deepmatcher
from autogoal.datasets import download_and_save, unpack, datapath

class DeepMatcherDataset:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.filename = pathlib.Path(url).name

    def load(self):
        save_dir = datapath(self.name)
        save_as = save_dir / self.filename

        if download_and_save(self.url, save_as, overwrite=False):
            unpack(str(save_as))

        train, validation, test = deepmatcher.data.process(
            path=save_dir,
            train='train.csv',
            validation='validation.csv',
            test='test.csv'
        )

        return train, validation, test
