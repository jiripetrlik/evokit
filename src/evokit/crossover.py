import numpy as np

def simulatedBinaryCrossover(parent1, parent2, child1, child2, nc = 2):
    size = len(parent1.values)
    exponent = 1 / (nc + 1)
    u = np.random.uniform(size = size)
    for i in range(size):
        if u[i] <= 0.5:
            bqi = (2 * u[i]) ** exponent
        else:
            bqi = (1 / 2 * (1 - u[i])) ** exponent

        value1 = 0.5 * (((1 + bqi) * parent1.values[i]) + ((1 - bqi) * parent2.values[i]))
        value2 = 0.5 * (((1 - bqi) * parent1.values[i]) + ((1 + bqi) * parent2.values[i]))

        if value1 < child1.minValues[i]:
            value1 = child1.minValues[i]
        if value1 > child1.maxValues[i]:
            value1 = child1.maxValues[i]
        if value2 < child2.minValues[i]:
            value2 = child2.minValues[i]
        if value2 > child2.maxValues[i]:
            value2 = child2.maxValues[i]

        child1.values[i] = value1
        child2.values[i] = value2
