from abc import ABC
import numpy as np

class Chromosome(ABC):
    def initialize(self):
        pass

class NumericChromosome(Chromosome):
    def __init__(self, size, minValues, maxValues):
        self.values = np.zeros(size)
        self.minValues = minValues
        self.maxValues = maxValues

    def initialize(self):
        self.values = np.random.uniform(self.minValues, self.maxValues, len(self.values))
