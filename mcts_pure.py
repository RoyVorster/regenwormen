import numpy as np
from copy import deepcopy, copy

from game.game import Game

class Node:
    def __init__(self, game, action):
        self.game, self.action = game, action # Store action for retrieving later
        self.parent, self.children = None, []

        self.reward, self.n_sims = 0, 1e-5
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
        scores = [child.score for child in self.children]
        if not len(scores):
            return None

        # Select highest score or random if multiple are the same
        idx = np.random.choice(np.where(scores == np.max(scores))[0])
        return self.children[idx]

    @property
    def best_action(self):
        return self.action if self.best_child is None else self.best_child.action

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    # Play single turn
    def play_out(self, n_turns=2):
        game = deepcopy(self.game)

        turn_prev, n_turns_played = self.game.turn, 0
        while not game.game_done and n_turns_played < n_turns:
            action = np.random.choice(game.get())
            game.do(action)

            if game.turn != turn_prev:
                n_turns_played += 1
                turn_prev = game.turn

        turn = game.turn
        return game.scores[turn] - self.game.scores[turn]

# Simplest possible pure MCTS implementation
class MCTS:
    def __init__(self, n_iter=100):
        self.n_iter = n_iter

    def set_root(self, game):
        self.root = Node(game, "")

    def single_iteration(self):
        path = [self.root]
        while True:
            best_child = path[-1].best_child
            if not best_child:
                break

            path.append(best_child)

        # Run playouts for children states for node
        node = path[-1]
        for action in node.game.get():
            new_game = deepcopy(node.game)
            new_game.do(action)

            # Create new node and play the game until the end
            new_node = Node(new_game, action)
            node.add_child(new_node)

            reward, turn = new_node.play_out(), new_node.game.turn

            # Backprop the rewards
            for node in path:
                node.reward += reward
                node.n_sims += 1

    def train(self, print_progress=False):
        for i in range(self.n_iter):
            if print_progress and (n_iter - i) % (self.n_iter/100) == 0:
                print(f"At {100*i/n_iter:.2f}%", end="\r")

            self.single_iteration()

    def play(self, game):
        # Train from this node outward
        self.set_root(game)
        self.train()

        return self.root.best_action

if __name__ == '__main__':
    game = Game(n_players=2)

    ian = MCTS(game) # In honor of the player of games
    ian.train()

    print(ian.root.children[0].children)
