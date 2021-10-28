from abc import ABC
import numpy as np

class Chromosome(ABC):
    def initialize(self):
        pass

class NumericChromosome(Chromosome):
    def __init__(self, size, minValues, maxValues, randomGenerator = None):
        self.values = np.zeros(size)
        self.minValues = minValues
        self.maxValues = maxValues

        if randomGenerator != None:
            self.randomGenerator = randomGenerator
        else:
            self.randomGenerator = np.random.default_rng()

    def initialize(self):
        self.values = self.randomGenerator.uniform(self.minValues, self.maxValues, len(self.values))
