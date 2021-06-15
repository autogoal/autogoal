import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

class God():
    """
    This dude pits any two agents against each other. It is used for the self play training and to simulate a game match.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: each is a function that takes a board as input and returns an action in that board
            game: Game object
            display: a function that takes a board as input and prints it 
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    @staticmethod
    def playGame(player1, player2, game, display=None, verbose=False):
        """
        Executes one episode of a game.

         Input:
            player 1,2: each is a function that takes a board as input and returns an action in that board
            game: Game object
            display: a function that takes a board as input and prints it 

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [player2, None, player1]
        currentPlayer = 1
        board = game.getInitBoard()
        it = 0
        while game.getGameEnded(board, currentPlayer) == 0:
            it += 1
            if verbose:
                assert display
                print("Turn ", str(it), "Player ", str(currentPlayer))
                display(board)
            
            action = players[currentPlayer + 1](game.getCanonicalForm(board, currentPlayer))
            
            log.debug('player: ' + str(currentPlayer) + 'selected action: ' + str(action))

            valids = game.getValidMoves(game.getCanonicalForm(board, currentPlayer), 1)

            if not valids[action]:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, currentPlayer = game.getNextState(board, currentPlayer, action)
        if verbose:
            assert display
            print("Game over: Turn ", str(it), "Result ", str(game.getGameEnded(board, 1)))
            display(board)
        return currentPlayer * game.getGameEnded(board, currentPlayer)

    @staticmethod
    def playMatch(num, player1, player2, game, display=None,verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

         Input:
            num: the total number of games to play
            player 1,2: each is a function that takes a board as input and returns an action in that board
            game: Game object
            display: a function that takes a board as input and prints it 

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena games (1)"):
            gameResult = God.playGame(player1, player2, game, display= display, verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        player1, player2 = player2, player1

        for _ in tqdm(range(num), desc="Arena games (2)"):
            gameResult = God.playGame(player1, player2, game, display = display, verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws