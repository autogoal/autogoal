import abc

class Game(abc.ABC):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractproperty
    def name(self):
        pass

    @abc.abstractmethod
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board
        """
        pass

    @abc.abstractmethod
    def getBoardSize(self):
        """
        Returns:
            (x,y): the board dimensions
        """
        pass

    @abc.abstractmethod
    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    @abc.abstractmethod
    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 if player1, -1 if player2)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    @abc.abstractmethod
    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize() with values: 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass

    @abc.abstractmethod
    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        pass

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board*player

    
    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    @abc.abstractmethod
    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass    