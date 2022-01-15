import numpy as np
from game_roll import Roll


MIN_DOM, MAX_DOM = 21, 36

# Actions
ROLL = "ROLL" # Roll dice
SELECT = "SELECT" # Select dice
TAKE = "TAKE" # Take domino
GIVE_UP = "GIVE UP" # Failure mode

class Action:
    def __init__(self, action, option=None):
        self.action, self.option = action, option

    def __repr__(self):
        return self.action + (f"(arg: {self.option})" if self.option is not None else "")

class Game:
    def __init__(self, n_players=2, n_dice=8):
        self.turn = 0
        self.player_stacks = [[] for _ in range(n_players)]
        self.board_stack = list(range(MIN_DOM, MAX_DOM + 1))

        # Initialize roll
        self.roll = Roll(n_dice=n_dice)

        # Game parameters
        self.n_players = n_players

    @property
    def own_stack(self):
        return self.player_stacks[self.turn]

    @property
    def top(self):
        return len(self.own_stack) and self.own_stack[-1]

    @property
    def tops(self):
        return [stack[-1] if len(stack) > 0 and i != self.turn else 0 for i, stack in enumerate(self.player_stacks)]

    ''' Perform an action '''
    def do(self, action: Action):
        if action.action == ROLL:
            self.roll.do()
        elif action.action == SELECT:
            die = action.option
            self.roll.select(die)
        elif action.action == TAKE:
            taken = self.roll.valid and self.take_domino(self.roll.total)
            if not taken:
                self.give_up()

            self.next()
        elif action.action == GIVE_UP:
            self.give_up()
            self.next()

    ''' Return all actions '''
    def get(self):
        if self.game_done():
            return []

        options = []

        # Rolling options
        if self.roll.dice_left > 0:
            if self.roll.ready:
                options.append(Action(ROLL))
            else:
                selectable = [s for s in self.roll.roll if s not in self.roll.total_roll]
                options.extend([Action(SELECT, option=s) for s in set(selectable)])

        # Domino options 
        if self.roll.valid and self.take_domino(self.roll.total, dry_run=True):
            options.append(Action(TAKE))

        return options if len(options) > 0 else [Action(GIVE_UP)]

    ''' Check game done '''
    def game_done(self):
        return len(self.board_stack) == 0

    ''' Get game winner '''
    def game_winner(self):
        totals = [sum(stack) for stack in self.player_stacks]
        return totals.index(max(totals))

    ''' Give up and return domino if necessary '''
    def give_up(self):
        if self.top:
            self.own_stack.pop()
            self.board_stack.append(self.top)

        self.board_stack.remove(max(self.board_stack))

    ''' Take domino '''
    def take_domino(self, domino, dry_run=False):
        return self.take_domino_from_board(domino, dry_run) or \
               self.take_domino_from_player(domino, dry_run)

    ''' Take domino from baord stack '''
    def take_domino_from_board(self, domino, dry_run=False):
        available = [d for d in self.board_stack if domino >= d]
        if len(available):
            if not dry_run:
                domino = max(available)
                self.own_stack.append(domino)
                self.board_stack.remove(domino)

            return True

        return False

    ''' Take domino from top of player stack '''
    def take_domino_from_player(self, domino, dry_run=False):
        if domino in self.tops:
            if not dry_run:
                player_idx = self.tops.index(domino)
                self.player_stacks[player_idx].pop()

            return True

        return False

    ''' Next player '''
    def next(self):
        self.turn = (self.turn + 1) % self.n_players
        self.roll.reset()


if __name__ == '__main__':
    game = Game(n_players=2)

    while True:
        options = game.get()
        if not len(options):
            break

        game.do(options[-1])

    print(game.game_winner())
