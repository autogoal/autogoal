from os import listdir
from autogoal.datasets import download_and_save, datapath, unpack

data_url = "https://github.com/AntonioJesus0398/speech-recognition-dataset/raw/master/speech-recognition-dataset.zip"

dataset = "speech-recognition-dataset"


def load():
    # Download the data
    fname = f"{dataset}.zip"
    file_path = datapath(fname)
    if not file_path.exists():
        print(f"Downloading {dataset}...")
        download_and_save(data_url, file_path, True)
        print("Done!\n")

    # Unpack the zip file
    dir_path = datapath(dataset)
    if not dir_path.exists():
        print(f"Unpacking {fname}...")
        unpack(fname)
        print("Done.\n")

    # build the training and test sets
    X_train, y_train, X_test, y_test = [], [], [], []
    for subfolder in listdir(str(dir_path)):
        base_path = f"{dir_path}/{subfolder}/"
        for audio_file in listdir(base_path)[:-1]:
            file_path = base_path + audio_file
            X_train.append(file_path)
            y_train.append(subfolder)
        test_audio_file = base_path + listdir(base_path)[-1]
        X_test.append(test_audio_file)
        y_test.append(subfolder)
    return X_train, y_train, X_test, y_test
