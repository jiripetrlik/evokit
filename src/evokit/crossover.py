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

class PartiallyMappedCrossover:
    def crossover(self, parent1, parent2, child1, child2):
        size = len(parent1.values)
        points = np.random.randint(low = 0, high = size, size = 2)
        points = np.sort(points)

        exchangedPart1 = parent1.values[points[0]:points[1]]
        exchangedPart2 = parent2.values[points[0]:points[1]]
        missing1 = np.setdiff1d(exchangedPart2, exchangedPart1)
        missing2 = np.setdiff1d(exchangedPart1, exchangedPart2)

        mapping1 = {}
        mapping2 = {}
        for i in range(points[0], points[1]):
            if parent2.values[i] in missing1:                
                mapping1[parent2.values[i]] = parent1.values[i]
            if parent1.values[i] in missing2:                
                mapping2[parent1.values[i]] = parent2.values[i]
        
        child1.values[0:points[0]] = parent1.values[0:points[0]]
        child2.values[0:points[0]] = parent2.values[0:points[0]]
        child1.values[points[0]:points[1]] = parent2.values[points[0]:points[1]]
        child2.values[points[0]:points[1]] = parent1.values[points[0]:points[1]]
        child1.values[points[1]:size] = parent1.values[points[1]:size]
        child2.values[points[1]:size] = parent2.values[points[1]:size]

        i = size - 1
        while len(mapping1) > 0:
            value = child1.values[i]
            if value in missing1:
                child1.values[i] = mapping1[value]
                del mapping1[value]
            i -= 1
        i = size - 1
        while len(mapping2) > 0:
            value = child2.values[i]
            if value in missing2:
                child2.values[i] = mapping2[value]
                del mapping2[value]
            i -= 1
