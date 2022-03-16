import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import reduce, partial

from mcts import MCTS, f_reward
from game.game import *

N_CPU_MAX = 10

# Random agent for evaluation
def play_random(game):
    return np.random.choice(game.get())

# Simple greedy agent for evaluation
def take_highest(actions):
    return reduce(lambda a, b: a if a.option > b.option else b, actions)

def play_greedy(game, select_max_sum=True):
    options = game.get()

    if len(options) == 1:
        return options[0]

    take_actions = [o for o in options if o.action == TAKE]
    if len(take_actions):
        return take_highest(take_actions)

    select_actions = [o for o in options if o.action == SELECT]
    if len(select_actions):
        highest = take_highest(select_actions)
        if highest.option == 6: return highest # Prioritize wurmpies

        if select_max_sum:
            v = np.array([sum(game.roll.roll == a.option)*a.option for a in select_actions])
            return select_actions[np.argmax(v)]

        return highest

    return np.random.choice(options)

def play_game(agents):
    game = Game(n_players=len(agents))
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
def evaluate(agents, n_games=30):
    def loop():
        idxs = np.random.choice(len(agents), len(agents))
        return np.array(play_game(np.array(agents)[idxs]))[idxs]

    all_scores = Parallel(n_jobs=N_CPU_MAX)(delayed(loop)() for _ in range(n_games))
    all_scores = np.array(all_scores).T

    # Plot results
    fig, (ax_1, ax_2) = plt.subplots(nrows=2)
    for i, agent_scores in enumerate(all_scores):
        ax_1.plot(np.cumsum(agent_scores), label=f"Agent {i}")

        rewards = [f_reward(s, all_scores[:, j]) for j, s in enumerate(agent_scores)]
        ax_2.plot(np.cumsum(rewards), label=f"Agent {i}")

    ax_1.grid()
    ax_1.legend()

    ax_2.grid()
    ax_2.legend()

    ax_1.set_title("Worm count")
    ax_2.set_title("Reward function")

    plt.show()

if __name__ == '__main__':
    agents = [MCTS(n_iter=10).play, partial(play_greedy, select_max_sum=False), partial(play_greedy, select_max_sum=True)]
    evaluate(agents, n_games=100)

