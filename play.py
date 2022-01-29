from mcts_pure import MCTS
from game.game import Game

import numpy as np

# Random ai for evaluation
def play_random(game):
    options = game.get()
    return np.random.choice(options)

def play_game(agents):
    game = Game(n_players=len(agents))

    ian = MCTS()
    while True:
        if game.game_done:
            break

        # Select agent and get move
        f = agents[game.turn]
        action = f(game)

        # Perform the action
        game.do(action)

if __name__ == '__main__':
    agents = [MCTS(n_iter=100).play, play_random]
    play_game(agents)

