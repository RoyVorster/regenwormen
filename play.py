import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import reduce

from mcts_pure import MCTS
from game.game import *

N_CPU_MAX = 10

# Random agent for evaluation
def play_random(game):
    return np.random.choice(game.get())

# Simple greedy agent for evaluation
def take_highest(actions):
    return reduce(lambda a, b: a if a.option > b.option else b, actions)

def play_greedy(game):
    options = game.get()

    if len(options) == 1:
        return options[0]

    take_actions = [o for o in options if o.action == TAKE]
    if len(take_actions):
        return take_highest(take_actions)

    select_actions = [o for o in options if o.action == SELECT]
    if len(select_actions):
        return take_highest(select_actions)

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

# Play a bunch of games and plot results
def evaluate(agents, n_games=100):
    all_scores = Parallel(n_jobs=N_CPU_MAX)(delayed(play_game)(agents) for _ in range(n_games))

    # Plot results
    all_scores = np.array(all_scores).T
    for i, agent_scores in enumerate(all_scores):
        plt.plot(np.cumsum(agent_scores), label=f"Agent {i}")

    plt.grid()
    plt.legend()

    plt.show()

if __name__ == '__main__':
    agents = [MCTS(n_iter=15).play, play_greedy, play_greedy, play_greedy, play_greedy, play_random]
    evaluate(agents)

