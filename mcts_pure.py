import numpy as np
from copy import deepcopy

from game.game import *

class Node:
    def __init__(self, game):
        self.game = game
        self.parent, self.children = None, []

        self.reward, self.n_sims = 0, 1
        self.c = np.sqrt(2) # Discovery parameter

    def __repr__(self):
        return f"Number of visits, reward: {self.n_sims}, {self.reward}"

    @property
    def score(self):
        return self.reward/self.n_sims + self.c*np.sqrt(np.log(self.parent.n_sims)/self.n_sims)

    @property
    def end_state(self):
        return self.game.game_done

    @property
    def best_child(self):
        scores = [child.score for child in self.children if not child.end_state]

        # Select highest score or random if multiple are the same
        idx = np.random.choice(np.where(scores == np.max(scores))[0])
        return self.children[idx]

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def play_out(self):
        game = deepcopy(self.game)
        while not game.game_done:
            action = np.random.choice(game.get())
            game.do(action)

        scores = game.scores
        reward = 1 if scores[self.game.turn] == max(scores) else 0

        return reward

# Simplest possible pure MCTS implementation
class MCTS:
    def __init__(self, game):
        self.root = Node(game)

    def single_iteration(self):
        path = [self.root]
        while len(path[-1].children):
            path.append(path[-1].best_child)

        # Run playouts for children states for node
        node = path[-1]
        for action in node.game.get():
            new_game = deepcopy(node.game)
            new_game.do(action)

            # Create new node and play the game until the end
            new_node = Node(new_game)
            node.add_child(new_node)

            reward, turn = new_node.play_out(), new_node.game.turn

            # Backprop the rewards
            for node in path:
                node.reward += reward*int(turn == node.game.turn) # Only give reward if we're the same player
                node.n_sims += 1

    def train(self, n_iter=5000):
        for i in range(1, n_iter):
            if (n_iter - i) % (n_iter/100) == 0:
                print(f"At {100*i/n_iter:.2f}%", end="\r")

            self.single_iteration()

if __name__ == '__main__':
    game = Game(n_players=2)

    ian = MCTS(game) # In honor of the player of games
    ian.train()

    print(ian.root.children)
