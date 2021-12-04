from abc import ABC
import numpy as np

class Chromosome(ABC):
    def initialize(self):
        pass

class ChromosomeFactory(ABC):
    def createChromosome(self):
        pass

class BinaryChromosome:
    def __init__(self, size):
        self.values = np.zeros(size, dtype=np.dtype(bool))

    def initialize(self):
        self.values = np.random.choice([True, False], size = len(self.values))

    def copyValues(self, originalChromosome):
        np.copyto(self.values, originalChromosome.values)

    def getInteger(self, startBit, stopBit):
        i = 0
        result = 0
        while startBit + i < stopBit:
            result += self.values[startBit + i] * (2 ** i)
            i += 1

        return result

    def getFloat(self, startBit, stopBit, min, max):
        intMax = 2 ** (stopBit - startBit) - 1
        intValue = self.getInteger(startBit, stopBit)
        result = ((intValue / intMax) * (max - min)) + min

        return result

class BinaryChromosomeFactory:
    def __init__(self, size):
        self.size = size

    def createChromosome(self):
        return BinaryChromosome(self.size)

class NumericChromosome(Chromosome):
    def __init__(self, size, minValues, maxValues):
        self.values = np.zeros(size)
        self.minValues = np.array(minValues)
        self.maxValues = np.array(maxValues)

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

class PermutationChromosome(Chromosome):
    def __init__(self, size):
        self.values = np.random.permutation(size)

    def initialize(self):
        self.values = np.random.permutation(len(self.values))

    def copyValues(self, originalChromosome):
        np.copyto(self.values, originalChromosome.values)

class PermutationChromosomeFactory(ChromosomeFactory):
    def __init__(self, size):
        self.size = size

    def createChromosome(self):
        return PermutationChromosome(self.size)
