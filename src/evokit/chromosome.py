from abc import ABC
import numpy as np

class Chromosome(ABC):
    def initialize(self):
        pass

class ChromosomeFactory(ABC):
    def createChromosome(self):
        pass

class NumericChromosome(Chromosome):
    def __init__(self, size, minValues, maxValues):
        self.values = np.zeros(size)
        self.minValues = minValues
        self.maxValues = maxValues

    def initialize(self):
        self.values = np.random.uniform(self.minValues, self.maxValues, len(self.values))

    def renormalize(self):
        self.values = np.maximum(self.values, self.minValues)
        self.values = np.minimum(self.values, self.maxValues)

    def copyValues(self, originalChromosome):
        np.copyto(self.values, originalChromosome.values)

class NumericChromosomeFactory(ChromosomeFactory):
    def __init__(self, size, minValues, maxValues):
        self.size = size
        self.minValues = minValues
        self.maxValues = maxValues

    def createChromosome(self):
        return NumericChromosome(self.size, self.minValues, self.maxValues)
