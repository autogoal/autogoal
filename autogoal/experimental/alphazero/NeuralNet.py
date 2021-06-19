from . import config
import numpy as np
import tensorflow as tf
import os

from tensorflow.python.keras.layers.core import Dropout, Reshape
from tensorflow.keras.models import *
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Flatten,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.optimizers import SGD, Adam


class NeuralNetwork:
    def __init__(self, reg_const, learning_rate, batch_size, epochs, game):
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.input_dim = game.getBoardSize()
        self.output_dim = game.getActionSize()
        self.game = game
        self.model = self._build_model()

    def predict(self, state_board):
        """
        Returns the predicted values of pi and v for the board
        """
        board = np.array([self.convertToModelInput(state_board)])
        pi, v = self.model.predict(board)

        return pi[0], v[0]

    def train(self, examples):  # examples is a list of tuples: [(board, pi, value)]
        """
        Trains the neural network with the provided examples
        """
        train_boards, train_pis, train_values = list(zip(*examples))

        train_boards = np.asarray(train_boards)
        train_pis = np.asarray(train_pis)
        train_values = np.asarray(train_values)
        self.model.fit(
            train_boards,
            [train_pis, train_values],
            batch_size=config.TRAIN_BATCH_SIZE,
            epochs=config.TRAIN_EPOCHS,
        )

    def fit(self, states, targets, verbose, validation_split):
        return self.model.fit(
            states,
            targets,
            epochs=self.epochs,
            verbose=verbose,
            validation_split=validation_split,
            batch_size=self.batch_size,
        )

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint"):
        filepath = os.path.join(folder, filename)
        print("----Save File Path-----   " + filepath)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.model.save(filepath + ".h5")

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint"):
        filepath = os.path.join(folder, filename)
        print("----Load File Path-----   " + filepath)
        if not os.path.exists(filepath + ".h5"):
            raise Exception("No model found in path")
        self.model = load_model(filepath + ".h5")

    def _build_model(self):
        main_input = Input(shape=self.input_dim, name="main_input")
        board_x, board_y = self.input_dim

        x_image = Reshape((board_x, board_y, 1))(
            main_input
        )  # batch_size  x board_x x board_y x 1
        h_conv1 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(config.NUM_CHANNELS, 3, padding="same")(x_image)
            )
        )  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(config.NUM_CHANNELS, 3, padding="same")(h_conv1)
            )
        )  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(config.NUM_CHANNELS, 3, padding="same")(h_conv2)
            )
        )  # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation("relu")(
            BatchNormalization(axis=3)(
                Conv2D(config.NUM_CHANNELS, 3, padding="valid")(h_conv3)
            )
        )  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(config.DROPOUT)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat)))
        )  # batch_size x 1024
        s_fc2 = Dropout(config.DROPOUT)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(512)(s_fc1)))
        )  # batch_size x 1024
        pi = Dense(self.output_dim, activation="softmax", name="pi")(
            s_fc2
        )  # batch_size x self.action_size
        v = Dense(1, activation="tanh", name="v")(s_fc2)  # batch_size x 1

        model = Model(inputs=main_input, outputs=[pi, v])
        model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(self.learning_rate),
        )

        return model

    def convertToModelInput(self, state):
        inputToModel = state
        inputToModel = np.reshape(inputToModel, self.input_dim)

        return inputToModel


def softmax_cross_entropy_with_logits(y_true, y_pred):

    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss
