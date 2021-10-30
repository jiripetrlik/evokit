import numpy as np

class SimulatedBinaryCrossover:
    def __init__(self, nc = 2):
        self.nc = nc
    
    def crossover(self, parent1, parent2, child1, child2):
        size = len(parent1.values)
        exponent = 1 / (self.nc + 1)
        u = np.random.uniform(size = size)
        for i in range(size):
            if u[i] <= 0.5:
                bqi = (2 * u[i]) ** exponent
            else:
                bqi = (1 / 2 * (1 - u[i])) ** exponent

            value1 = 0.5 * (((1 + bqi) * parent1.values[i]) + ((1 - bqi) * parent2.values[i]))
            value2 = 0.5 * (((1 - bqi) * parent1.values[i]) + ((1 + bqi) * parent2.values[i]))

            child1.renormalize()
            child2.renormalize()

            child1.values[i] = value1
            child2.values[i] = value2
