
#### SELF PLAY
NUM_ITERS = 10		# Total number of training iterations. The player's training algorithm will be executed a total of NUM_ITERS times.
TEMP_THRESHOLD = 15
EPISODES = 50       # Number of complete self play games to simulate during a new iteration.
MCTS_SIMS = 50      # Number of games moves for MCTS to simulate.
QUEUE_LEN = 20000	# Max number of train examples that can be returned by one episode of self play
MEMORY_SIZE = 300	# Max ammount of train example sets that can be provided to the net after the self play episodes are done
					## NOTE: when the self play episodes are completed, a list of size <= MEMORY_SIZE * QUEUE_LEN is passed to the nnet as training input
					## each element in that list is an example of the form (board, pi, v)
TIME_LIMIT = 900
CPUCT = 1
EPSILON = 0.2
CHECKPOINT = './checkpoints'
ARENA_GAMES = 40     # Number of games to play during arena play to determine if new net will be accepted.
UPDATE_THRESHOLD = 0.6  # During arena playoff, the new neural net will be accepted if threshold or more of games are won.


#### NEURAL NET TRAINING
TRAIN_BATCH_SIZE = 64
TRAIN_EPOCHS = 10
NUM_CHANNELS=512
DROPOUT = 0.3
LEARNING_RATE = 0.0001
REG_CONST = 0.0001
MOMENTUM = 0.9

HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	]
