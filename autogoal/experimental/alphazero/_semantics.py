from autogoal.kb import SemanticType
from .GameInterface import Game

# To train an agent using reiforcement learning, we need to give
# the agent a context. This context represents all the possible 
# information that the agent might possess in a given moment.
# This type of information could include resources location,
# resource distance, path reliability, travel time, amount of
# resources, penalties for take a resource, etc, etc.

# The best way to represent this context, is through the 
# concept of an Environment (to simplify things, lets restrict this
# to simply a 2D environment), i.e, a Matrix of Discrete values.

# Every value of the Matrix represents a resource, except for the
# 0, which might indicate a "NULL" position. Resources could be
# anything, X or O in a Tic Tac Toe board game, Pawns in a Chess
# board, or creeps in a Dota2 Map.

class Environment(SemanticType):
    pass

class GameStructure(Environment):
    """The Game structure defines an Environment for 2-Players Game board
       with perfect information available. This should match our GameInterface
       which allow the implementatio of arbitrary games.

    Args:
        Environment ([type]): [description]
    """
    @classmethod
    def _match(cls, x) -> bool:
        return issubclass(x, Game)


# For games, the purpouse of a AI algorithm is to find the "best"
# possible player for the given game. So we need a way to define 
# the structure of a player

class Player(SemanticType):
    @classmethod
    def _match(cls, x) -> bool:
        return hasattr(x, "play")