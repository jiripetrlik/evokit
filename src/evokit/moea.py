import numpy as np

class WeightedSumFitness:
    def __init__(self, weights, fitnessFunctions):
        self.weights = np.copy(weights)
        self.fitnessFunctions = list(fitnessFunctions)

    def __call__(self, *args, **kwds):
        fitnessValues = [f(args[0]) for f in self.fitnessFunctions]
        return np.sum(self.weights * fitnessValues)
