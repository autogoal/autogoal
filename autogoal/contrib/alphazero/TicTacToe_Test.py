from .GameManager import God
from .ramdomAgent import RandomPlayer
from .Agent import AlphaZeroPlayer
from .TicTacToeGame import TicTacToeGame

game = TicTacToeGame()

player1 = AlphaZeroPlayer(game) #** create a new alphazero agent\ 
                                #**from scratch and train it

#**uncomment next line and comment previous line to load a pretrained player
#player1 = AlphaZeroPlayer.load(game)


player2 = RandomPlayer(game)

God.playMatch(2, player1.play, player2.play, game, display= game.display, verbose=True)

