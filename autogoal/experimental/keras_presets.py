from autogoal.utils import nice_repr, PreemptiveStopException
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
from autogoal.contrib.gensim import Word2VecRandom

from autogoal.kb import (
    AlgorithmBase,
    VectorCategorical,
    Supervised,
    Seq,
    Word,
    algorithm,
    VectorContinuous,
)
from autogoal.grammar import CategoricalValue, DiscreteValue
import numpy as np
import time

KERAS_TIMEOUT = None


def set_keras_timeout_data(timeout, min_epochs, cross_validation_steps):
    global KERAS_TIMEOUT
    KERAS_TIMEOUT = {
        "timeout": timeout,
        "min_epochs": min_epochs,
        "cross_validation_steps": cross_validation_steps,
    }


class PreemptiveTimeout(Callback):
    def __init__(self, timeout, init_time):
        self.timeout_data = timeout
        self.fit_init_time = init_time
        self.callback_init_time = time.time()
        self.prep_time = self.callback_init_time - self.fit_init_time

    def on_epoch_end(self, epoch, logs={}):
        if self.timeout_data != None:
            # Getting total time and time from starting the callback
            total_time = time.time() - self.fit_init_time
            callback_time = time.time() - self.callback_init_time

            # Read timeout_data
            timeout = self.timeout_data["timeout"]
            min_evals = self.timeout_data["min_epochs"]
            cross_validation_steps = self.timeout_data["cross_validation_steps"]

            # define max times
            max_epoch_time = (timeout - self.prep_time) / (
                min_evals * cross_validation_steps
            )
            max_eval_time = timeout / cross_validation_steps

            # First epoch tends to be a bit slower, hence the special case for it
            epoch_mean_time = (
                callback_time / (epoch + 1)
                if epoch is not 0
                else (callback_time * 1.1) / (epoch + 1)
            )

            if epoch_mean_time > max_epoch_time:
                raise PreemptiveStopException(
                    f"Preemptive stop: mean epoch time of {epoch_mean_time} is higher than expected {max_epoch_time}"
                )

            if total_time > max_eval_time:
                raise PreemptiveStopException(
                    f"Preemptive stop: total evaluation time of {total_time} is higher than expected {max_eval_time}"
                )


class KerasSentenceClassifier(AlgorithmBase):
    def __init__(
        self,
        optimizer,
        epochs=10,
        early_stop=3,
        validation_split=0.1,
        embedding=None,
        padding_type="Max",
    ):
        self.optimizer = optimizer
        self._epochs = epochs
        self._mode = "train"
        self._graph = None
        self._validation_split = validation_split
        self._early_stop = early_stop
        self.embedding = embedding
        self.padding_type = padding_type
        self.oov = "<OOV>"
        self.pad = "<PAD>"

        self.padding_length = 0
        self.model = None
        self.vocab_size = 2
        self.word2index = {self.pad: 0, self.oov: 1}
        self._timeout_data = KERAS_TIMEOUT

    def run(
        self, X: Seq[Seq[Word]], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        if self._mode == "train":
            self.fit(X, y)
            return y
        else:
            return self.predict(X)

    def train(self,):
        self._mode = "train"

    def eval(self,):
        self._mode = "eval"

    def disable_preemptive_stop(self,):
        self._timeout_data = None

    def build_model(self, n_classes):
        pass

    def fit(self, X, y):
        self._init_time = time.time()
        seqs = self.build_sequences(X, True)
        self.build_model(len(set(y)))

        self.model.summary()
        self.model.fit(
            x=seqs,
            y=np.asarray(y, dtype="int32"),
            epochs=self._epochs,
            callbacks=[
                EarlyStopping(patience=self._early_stop, restore_best_weights=True),
                TerminateOnNaN(),
                PreemptiveTimeout(self._timeout_data, self._init_time),
            ],
            validation_split=self._validation_split,
            verbose=1,
        )

    def build_sequences(self, docs, train):
        size = 2
        embeddings = []
        seqs = []
        max_size = 0
        min_size = np.inf
        acc_size = 0

        if train:
            self.vocab_size = 2
            self.word2index = {self.pad: 0, self.oov: 1}
            embeddings = [self.embedding.run(self.pad), self.embedding.run(self.oov)]
        for doc in docs:
            seq_doc = []
            for word in doc:
                if word not in self.word2index:
                    if train:
                        self.word2index[word] = size
                        emb = self.embedding.run(word)
                        embeddings.append(emb) if np.any(emb) else embeddings.append(
                            (np.random.random_sample(emb.shape) - 0.5) / 10
                        )
                        size += 1
                    else:
                        word = self.oov
                seq_doc.append(self.word2index[word])
            seqs.append(seq_doc)
            if train:
                max_size = max(max_size, len(seq_doc))
                min_size = min(min_size, len(seq_doc))
                acc_size += len(seq_doc)
        if train:
            self.padding_length = self._find_padding_length(
                min_size, max_size, acc_size, len(seqs)
            )
            self.vocab_size = size
            self.init_weights = np.asarray(embeddings, dtype="float32")
        return pad_sequences(
            seqs, int(self.padding_length), truncating="post", padding="pre"
        )

    def _find_padding_length(self, min_size, max_size, acc_size, docs_count):
        length_getters = {
            "Max": max_size,
            "Min": min_size,
            "Mean": acc_size / docs_count,
            "Min2Mean": ((acc_size / docs_count) + min_size) / 2,
            "Mean2Max": ((acc_size / docs_count) + max_size) / 2,
        }
        return length_getters[self.padding_type]

    def predict(self, X):
        seqs = self.build_sequences(X, False)
        return np.argmax(self.model.predict(seqs), axis=-1)


@nice_repr
class LSTMClassifier(KerasSentenceClassifier):
    def __init__(
        self,
        optimizer: CategoricalValue("sgd", "adam", "rmsprop"),
        lstm_size: DiscreteValue(32, 512),
        dense_layers: DiscreteValue(0, 4),
        dense_layer_size: DiscreteValue(32, 512),
        dense_layer_activation: CategoricalValue(
            "elu",
            "exponential",
            "linear",
            "relu",
            "selu",
            "sigmoid",
            "softmax",
            "softplus",
            "softsign",
            "tanh",
        ),
        embedding: algorithm(Word, VectorContinuous, always_include=[Word2VecRandom]),
        padding_type: CategoricalValue("Max", "Min", "Mean", "Min2Mean", "Mean2Max"),
    ) -> None:
        self.lstm_size = int(lstm_size)
        self.dense_layers = int(dense_layers)
        self.dense_layer_size = int(dense_layer_size)
        self.dense_layer_activation = dense_layer_activation
        super().__init__(optimizer, embedding=embedding, padding_type=padding_type)

    def build_model(self, n_classes):
        self.model = Sequential()
        self.model.add(
            Embedding(
                self.vocab_size,
                self.init_weights.shape[-1],
                weights=[self.init_weights],
            )
        )
        self.model.add(LSTM(self.lstm_size))
        for _ in range(self.dense_layers):
            self.model.add(
                Dense(self.dense_layer_size, activation=self.dense_layer_activation)
            )
        self.model.add(Dense(n_classes, activation="softmax"))
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )


@nice_repr
class BiLSTMClassifier(KerasSentenceClassifier):
    def __init__(
        self,
        optimizer: CategoricalValue("sgd", "adam", "rmsprop"),
        lstm_size: DiscreteValue(32, 512),
        dense_layers: DiscreteValue(0, 4),
        dense_layer_size: DiscreteValue(32, 512),
        dense_layer_activation: CategoricalValue(
            "elu",
            "exponential",
            "linear",
            "relu",
            "selu",
            "sigmoid",
            "softmax",
            "softplus",
            "softsign",
            "tanh",
        ),
        embedding: algorithm(Word, VectorContinuous, always_include=[Word2VecRandom]),
        padding_type: CategoricalValue("Max", "Min", "Mean", "Min2Mean", "Mean2Max"),
    ) -> None:
        self.lstm_size = int(lstm_size)
        self.dense_layers = int(dense_layers)
        self.dense_layer_size = int(dense_layer_size)
        self.dense_layer_activation = dense_layer_activation
        super().__init__(optimizer, embedding=embedding, padding_type=padding_type)

    def build_model(self, n_classes):
        self.model = Sequential()
        self.model.add(
            Embedding(
                self.vocab_size,
                self.init_weights.shape[-1],
                weights=[self.init_weights],
            )
        )
        self.model.add(Bidirectional(LSTM(self.lstm_size)))
        for _ in range(self.dense_layers):
            self.model.add(
                Dense(self.dense_layer_size, activation=self.dense_layer_activation)
            )
        self.model.add(Dense(n_classes, activation="softmax"))
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )


@nice_repr
class StackedLSTMClassifier(KerasSentenceClassifier):
    def __init__(
        self,
        optimizer: CategoricalValue("sgd", "adam", "rmsprop"),
        lstm_layers: DiscreteValue(2, 5),
        lstm_size: DiscreteValue(32, 512),
        dense_layers: DiscreteValue(0, 4),
        dense_layer_size: DiscreteValue(32, 512),
        dense_layer_activation: CategoricalValue(
            "elu",
            "exponential",
            "linear",
            "relu",
            "selu",
            "sigmoid",
            "softmax",
            "softplus",
            "softsign",
            "tanh",
        ),
    ) -> None:
        self.lstm_layers = int(lstm_layers)
        self.lstm_size = int(lstm_size)
        self.dense_layers = int(dense_layers)
        self.dense_layer_size = int(dense_layer_size)
        self.dense_layer_activation = dense_layer_activation
        super().__init__(optimizer)

    def build_model(self, shape, n_classes):
        self.model = Sequential()
        for _ in range(self.lstm_layers - 1):
            self.model.add(
                LSTM(self.lstm_size, input_shape=shape[-2:], return_sequences=True)
            )
        self.model.add(LSTM(self.lstm_size, input_shape=shape[-2:]))
        for _ in range(self.dense_layers):
            self.model.add(
                Dense(self.dense_layer_size, activation=self.dense_layer_activation)
            )
        self.model.add(Dense(n_classes, activation="softmax"))
        self.model.compile(optimizer="adam", loss="mse")


@nice_repr
class StackedBiLSTMClassifier(KerasSentenceClassifier):
    def __init__(
        self,
        optimizer: CategoricalValue("sgd", "adam", "rmsprop"),
        lstm_layers: DiscreteValue(2, 5),
        lstm_size: DiscreteValue(32, 512),
        dense_layers: DiscreteValue(0, 4),
        dense_layer_size: DiscreteValue(32, 512),
        dense_layer_activation: CategoricalValue(
            "elu",
            "exponential",
            "linear",
            "relu",
            "selu",
            "sigmoid",
            "softmax",
            "softplus",
            "softsign",
            "tanh",
        ),
    ) -> None:
        self.lstm_layers = int(lstm_layers)
        self.lstm_size = int(lstm_size)
        self.dense_layers = int(dense_layers)
        self.dense_layer_size = int(dense_layer_size)
        self.dense_layer_activation = dense_layer_activation
        super().__init__(optimizer)

    def build_model(self, shape, n_classes):
        self.model = Sequential()
        for _ in range(self.lstm_layers - 1):
            self.model.add(
                Bidirectional(
                    LSTM(self.lstm_size, input_shape=shape[-2:], return_sequences=True)
                )
            )
        self.model.add(Bidirectional(LSTM(self.lstm_size, input_shape=shape[-2:])))
        for _ in range(self.dense_layers):
            self.model.add(
                Dense(self.dense_layer_size, activation=self.dense_layer_activation)
            )
        self.model.add(Dense(n_classes, activation="softmax"))
        self.model.compile(optimizer="adam", loss="mse")

