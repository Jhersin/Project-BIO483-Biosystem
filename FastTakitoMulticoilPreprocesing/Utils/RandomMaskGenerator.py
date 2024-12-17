import numpy as np
import matplotlib.pyplot as plt
from Utils.subsample import RandomMask

class RandomMask:
    def __init__(self, rng=None):
        #Set up random number generator
        self.rng = rng if rng is not None else np.random.default_rng()

    def accel_mask(self, n, acceleration, offset, num_low_frequencies):
        # Calculate the probability of selecting each column. N/acceleration - num_low_frequencies is the number of columns
        prob = (n / acceleration - num_low_frequencies) / (n - num_low_frequencies)
        print(f'prob {n / acceleration - num_low_frequencies}')
        # Create the mask by randomly selecting columns with probability 'prob'
        mask = self.rng.uniform(size=n) < prob
        # Ensure low-frequency columns in the center are always selected
        mask[offset:offset + num_low_frequencies] = True
        return mask