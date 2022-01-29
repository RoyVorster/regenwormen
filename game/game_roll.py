import numpy as np

# Single roll (6 = wurmpie)
class Roll:
    def __init__(self, n_dice=8):
        self.reset()
        self.n_dice = n_dice

    @property
    def dice_left(self):
        return self.n_dice - len(self.total_roll)

    @property
    def valid(self):
        return 6 in self.total_roll

    @property
    def total(self):
        if not self.valid:
            return 0

        return sum([min(5, d) for d in self.total_roll])

    def reset(self):
        self.roll, self.total_roll = [], []
        self.ready = True

    def do(self):
        self.ready = False
        self.roll = np.random.randint(1, 7, (self.dice_left, ))
        return self.roll

    def select(self, die):
        self.ready = True
        self.total_roll.extend(self.roll[self.roll == die])
