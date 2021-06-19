import numpy as np
import time

from collections import deque
from random import shuffle, sample
from tqdm import tqdm

from .MCTS import MCTS
from .GameManager import God
from .config import *
from .NeuralNet import NeuralNetwork


class AlphaZeroPlayer:
    def __init__(self, game):
        self.game = game

        self.net = NeuralNetwork(REG_CONST, LEARNING_RATE, TRAIN_BATCH_SIZE, TRAIN_EPOCHS, self.game)
        self.agent = AlphaZeroAgent(self.game, self.net)

        print("Creating a new player. Training it")
        self.agent.train()

    def play(self, board):
        return self.agent.play(board)

    @staticmethod
    def load(game):
        try:
            net = NeuralNetwork(REG_CONST, LEARNING_RATE, TRAIN_BATCH_SIZE, TRAIN_EPOCHS, game)
            net.load_checkpoint(folder=game.name + "_trainedPlayer/", filename="best")
            return AlphaZeroAgent(game, net)
        except Exception:
            print("Unable to load. No pretrained model found")


class AlphaZeroAgent:
    """
    This does the self-play + learning
    """

    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet
        self.mcts = MCTS(self.game, self.nnet)
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False
        self.save_folder = game.name + "/"

    def executeRound(self):
        """
        This runs one round of self play, starting with player 1. The game is played untill it ends.

        It uses a temp = 1 if roundStep < tempThreshold, and temp = 0 thereafter

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currentPlayer, pi, v)
                        pi is the MCTS informed policy vector, v is +1 if the player won the game, else -1
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.currentPlayer = 1
        roundStep = 0

        while True:
            roundStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.currentPlayer)

            temp = int(roundStep < TEMP_THRESHOLD)
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)

            for b, p in sym:
                trainExamples.append([b, self.currentPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.currentPlayer = self.game.getNextState(
                board, self.currentPlayer, action
            )

            r = self.game.getGameEnded(board, self.currentPlayer)

            if r != 0:
                return [
                    (x[0], x[2], r * ((-1) ** (x[1] != self.currentPlayer)))
                    for x in trainExamples
                ]

    def train(
        self,
        num_iters=NUM_ITERS,
        time_limit=TIME_LIMIT,
        queue_len=QUEUE_LEN,
        episodes=EPISODES,
        memory_size=MEMORY_SIZE,
        arena_games=ARENA_GAMES,
        update_threshold=UPDATE_THRESHOLD
    ):
        """
        Performs numIters iterations with numRounds rounds of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        start_time = time.time()
        time_lapsed = 0
        i = 1
        while i < num_iters + 1 and time_lapsed < time_limit:
        #for i in range(1, num_iters + 1):
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=queue_len)

                for _ in tqdm(range(episodes), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet)
                    iterationTrainExamples += self.executeRound()

                self.trainExamplesHistory.append(iterationTrainExamples)

            # if len(self.trainExamplesHistory) > memory_size:
            #    self.trainExamplesHistory.pop(0)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            if len(trainExamples) > memory_size:
                trainExamples = sample(trainExamples, memory_size)

            # training new network
            self.nnet.save_checkpoint(
                folder=CHECKPOINT + self.save_folder, filename="temp"
            )

            self.pnet.load_checkpoint(
                folder=CHECKPOINT + self.save_folder, filename="temp"
            )

            pmcts = MCTS(self.game, self.pnet)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet)

            # pittin new net vs old one
            pwins, nwins, draws = God.playMatch(
                arena_games,
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game
            )

            if (
                pwins + nwins == 0 or float(nwins) / (pwins + nwins) < update_threshold
            ):  # the new net gets discarded
                self.nnet.load_checkpoint(
                    folder=CHECKPOINT + self.save_folder, filename="temp"
                )

            else:  # the new net is better and gets accepted
                self.nnet.save_checkpoint(
                    folder=CHECKPOINT + self.save_folder,
                    filename=self.getCheckpointFile(i),
                )
                self.nnet.save_checkpoint(
                    folder=self.game.name + "_trainedPlayer/", filename="best"
                )
            
            i += 1
            time_lapsed = time.time() - start_time

    def play(self, board):
        return np.argmax(self.mcts.getActionProb(board, temp=0))

    def getCheckpointFile(self, iteration):
        return "checkpoint_" + str(iteration)
