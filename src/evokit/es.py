import numpy as np

class OnePlusOneObserver:
    def __init__(self, iterations):
        self.minFitness = np.zeros(iterations)
        self.improved = np.zeros(iterations, dtype=np.bool_)
    
    def update(self, iteration, fitnessValue, improved):
        self.minFitness[iteration] = fitnessValue
        self.improved[iteration] = improved

def onePlusOneES(fitness, size, sd, iterations):
    vector1 = np.random.normal(scale = sd, size = size)
    bestFintess = fitness(vector1)
    observer = OnePlusOneObserver(size)
    for iter in range(iterations):
        vector2 = vector1 + np.random.normal(scale = sd, size = size)
        newFitness = fitness(vector2)
        improved = False
        if newFitness <= bestFintess:
            improved = True
            bestFintess = newFitness
            vector1 = vector2

        observer.update(iter, bestFintess, improved)
        
    results = {
        "fitness": bestFintess,
        "solution": vector1,
        "observer": observer
    }

    return results
