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
    def __map(self, value, mapping):
        while value in mapping:
            value = mapping[value]

        return value

    def crossover(self, parent1, parent2, child1, child2):
        size = len(parent1.values)
        points = np.random.randint(low = 0, high = size, size = 2)
        points = np.sort(points)

        mapping1 = {}
        mapping2 = {}
        for i in range(points[0], points[1]):
            mapping1[parent2.values[i]] = parent1.values[i]
            mapping2[parent1.values[i]] = parent2.values[i]
        
        child1.values[0:points[0]] = parent1.values[0:points[0]]
        child2.values[0:points[0]] = parent2.values[0:points[0]]
        child1.values[points[0]:points[1]] = parent2.values[points[0]:points[1]]
        child2.values[points[0]:points[1]] = parent1.values[points[0]:points[1]]
        child1.values[points[1]:size] = parent1.values[points[1]:size]
        child2.values[points[1]:size] = parent2.values[points[1]:size]

        uniqueResult1 = np.unique(child1.values, return_counts=True)
        duplicated1 = uniqueResult1[0][uniqueResult1[1] > 1]
        for i in range(size - 1, 0, -1):
            value = child1.values[i]
            if value in duplicated1:
                child1.values[i] = self.__map(value, mapping1)
                duplicated1 = np.setdiff1d(duplicated1, value)
        
        uniqueResult2 = np.unique(child2.values, return_counts=True)
        duplicated2 = uniqueResult2[0][uniqueResult2[1] > 1]
        for i in range(size - 1, 0, -1):
            value = child2.values[i]
            if value in duplicated2:
                child2.values[i] = self.__map(value, mapping2)
                duplicated2 = np.setdiff1d(duplicated2, value)
