import pathlib
import deepmatcher
import tempfile
import csv
from autogoal.datasets import download_and_save, unpack, datapath

class DeepMatcherDataset:
    def __init__(self, name, url):
        self.name = name
        self.url = url

        purl = pathlib.Path(url)
        self.filename = purl.name
        self.stem = purl.stem

    def load(self):
        save_dir = datapath(self.name)

        if not save_dir.exists():
            save_dir.mkdir()

        save_as = save_dir / self.filename

        print('Dowloading')
        if download_and_save(self.url, save_as, overwrite=False):
            print('Downloaded ... Unpacking')
            unpack(str(save_as))
            print('Unpacked')

        data_dir = save_dir / self.stem / 'exp_data'
        tr, va, te = self._load_full_data(data_dir)

        train, validation, test = deepmatcher.data.process(
            path=data_dir,
            train=tr.name,
            validation=va.name,
            test=te.name,
            ignore_columns=('left_id', 'right_id'),
            left_prefix='left_',
            right_prefix='right_',
            label_attr='label',
            id_attr='id',
            embeddings='wiki.simple.vec'
        )

        return train, validation, test

    def _load_full_data(self, data_dir):
        csv_writer = lambda: tempfile.NamedTemporaryFile('w+t')

        d = {
            'train': csv_writer(),
            'valid': csv_writer(),
            'test': csv_writer(),
        }

        table_a = open(data_dir / 'tableA.csv', newline='')
        table_b = open(data_dir / 'tableB.csv', newline='')

        ha, *ra = list(csv.reader(table_a))
        hb, *rb = list(csv.reader(table_b))

        assert ha == hb

        left_h = list(map(lambda x: 'left_' + x, ha))
        right_h = list(map(lambda x: 'right_' + x, hb))

        for name in ['train', 'valid', 'test']:
            file = name + '.csv'

            with open(data_dir / file, newline='') as f:
                reader = csv.reader(f)
                next(reader)

                w = csv.writer(d[name])
                w.writerow(['id', 'label'] + left_h + right_h)

                id = 0

                for left, right, label in reader:
                    left = int(left)
                    right = int(right)
                    w.writerow([id, label] + ra[left] + rb[right])
                    id += 1

        table_a.close()
        table_b.close()

        return d['train'], d['valid'], d['test']

