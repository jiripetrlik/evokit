import numpy as np

class OnePointCrossover:
    def crossover(self, parent1, parent2, child1, child2):
        size = len(parent1.values)

        point = np.random.randint(0, size)
        for i in range(point):
            child1.values[i] = parent1.values[i]
            child2.values[i] = parent2.values[i]
        for i in range(point, size):
            child1.values[i] = parent2.values[i]
            child2.values[i] = parent1.values[i]

class UniformCrossover:
    def crossover(self, parent1, parent2, child1, child2):
        size = len(parent1.values)

        r = np.random.uniform(size = size)
        for i in range(size):
            if r[i] < 0.5:
                child1.values[i] = parent1.values[i]
                child2.values[i] = parent2.values[i]
            else:
                child1.values[i] = parent2.values[i]
                child2.values[i] = parent1.values[i]

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
