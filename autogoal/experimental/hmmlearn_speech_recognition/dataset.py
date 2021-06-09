from os import listdir

root_dir = './autogoal/experimental/hmmlearn_speech_recognition'

def load():
    X_train, y_train, X_test, y_test = [], [], [], []
    for subfolder in listdir(f'{root_dir}/data'):
        base_path = f'{root_dir}/data/{subfolder}/'
        for audio_file in listdir(base_path)[:-1]:
            file_path = base_path + audio_file
            X_train.append(file_path)
            y_train.append(subfolder)
        test_audio_file = base_path + listdir(base_path)[-1]
        X_test.append(test_audio_file)
        y_test.append(subfolder)
    return X_train, y_train, X_test, y_test