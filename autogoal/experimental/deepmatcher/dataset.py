import pathlib
import deepmatcher
import tempfile
import csv
from autogoal.datasets import download_and_save, unpack, datapath


def generate_word_vector(file_path):
    import fasttext

    model = fasttext.train_unsupervised(file_path)
    with open(pathlib.Path(file_path).parent / "word.vec", "w") as f:
        for word in model.get_words():
            v = model.get_word_vector(word)
            f.write(word)
            for x in v:
                f.write(f" {x}")
            f.write("\n")


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

        if not save_as.exists() and download_and_save(
            self.url, save_as, overwrite=False
        ):
            print("Downloaded ... Unpacking")
            unpack(str(save_as))
            print("Unpacked")

        data_dir = save_dir / self.stem / "exp_data"
        tr, va, te = self._load_full_data(data_dir)

        def fix_type_issue(t):  # https://github.com/anhaidgroup/deepmatcher/issues/3
            for i in range(len(t[0])):
                if t[0][i] == "left_type":
                    t[0][i] = "left_entity_type"
                if t[0][i] == "right_type":
                    t[0][i] = "right_entity_type"

        fix_type_issue(tr)
        fix_type_issue(va)
        fix_type_issue(te)

        headers = tr.pop(0)
        assert headers == va.pop(0)
        assert headers == te.pop(0)
        tr += va

        def get_X_y(table):
            X, y = [], []
            for row in table:
                y.append(row.pop(1))
                X.append(row)
            return X, y

        X_train, y_train = get_X_y(tr)
        X_test, y_test = get_X_y(te)

        return headers, X_train, y_train, X_test, y_test

    def _load_full_data(self, data_dir):
        d = {
            "train": [],
            "valid": [],
            "test": [],
        }

        table_a = open(data_dir / "tableA.csv", newline="")
        table_b = open(data_dir / "tableB.csv", newline="")

        ha, *ra = list(csv.reader(table_a))
        hb, *rb = list(csv.reader(table_b))

        assert ha == hb

        left_h = list(map(lambda x: "left_" + x, ha))
        right_h = list(map(lambda x: "right_" + x, hb))

        for name in ["train", "valid", "test"]:
            file = name + ".csv"

            with open(data_dir / file, newline="") as f:
                reader = csv.reader(f)
                next(reader)

                d[name].append(["id", "label"] + left_h + right_h)

                id = 0

                for left, right, label in reader:
                    left = int(left)
                    right = int(right)
                    d[name].append([str(id), label] + ra[left] + rb[right])
                    id += 1

        table_a.close()
        table_b.close()

        return d["train"], d["valid"], d["test"]


if __name__ == "__main__":
    from autogoal.experimental.deepmatcher import DATASETS

    test_name = "Fodors-Zagats"
    dataset = DeepMatcherDataset(test_name, DATASETS[test_name])
    headers, X_train, y_train, X_test, y_test = dataset.load()
