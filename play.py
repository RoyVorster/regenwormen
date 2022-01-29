import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from mcts_pure import MCTS
from game.game import Game

N_CPU_MAX = 10

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

    return game.scores

# Play a bunch of games and store results
def evaluate(agents, n_games=50):
    all_scores = Parallel(n_jobs=min(n_games, N_CPU_MAX))(delayed(play_game)(agents) for _ in range(n_games))

    # Plot results
    all_scores = np.array(all_scores).T
    for i, agent_scores in enumerate(all_scores):
        plt.plot(np.cumsum(agent_scores), label=f"Agent {i}")

    plt.grid()
    plt.legend()

    plt.show()

if __name__ == '__main__':
    agents = [play_random, MCTS(n_iter=10).play]
    evaluate(agents)

