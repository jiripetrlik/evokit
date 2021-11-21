import numpy as np
import matplotlib.pyplot as plt

class OnePlusOneObserver:
    def __init__(self, iterations):
        self.minFitness = np.zeros(iterations)
        self.improved = np.zeros(iterations, dtype=np.bool_)
    
    def update(self, iteration, fitnessValue, improved):
        self.minFitness[iteration] = fitnessValue
        self.improved[iteration] = improved

    def plot(self):
        plt.plot(self.minFitness, label = "Fitness")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()

def onePlusOneES(fitness, size, sd, iterations):
    vector1 = np.random.normal(scale = sd, size = size)
    bestFintess = fitness(vector1)
    observer = OnePlusOneObserver(iterations)
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

class miPlusOneObserver:
    def __init__(self, iterations):
        self.minFitness = np.zeros(iterations)
        self.meanFitness = np.zeros(iterations)
        self.maxFitness = np.zeros(iterations)
        self.improved = np.zeros(iterations, dtype=np.bool_)

    def update(self, iteration, parentsFitness, improved):
        self.minFitness[iteration] = np.min(parentsFitness)
        self.meanFitness[iteration] = np.mean(parentsFitness)
        self.maxFitness[iteration] = np.max(parentsFitness)
        self.improved[iteration] = improved

def miPlusOneES(fitness, size, mi, iterations):
    parents = np.random.normal(size=(mi, size))
    parentsSD = np.random.normal(size=(mi, size))
    parentsFitness = np.apply_along_axis(fitness, 1, parents)
    worstSolution = np.argmax(parentsFitness)
    worstFitness = parentsFitness[worstSolution]
    observer = miPlusOneObserver(iterations)

    for iter in range(iterations):
        sample1 = np.random.randint(low = 0, high = mi, size = size)
        sample2 = np.random.randint(low = 0, high = mi, size = size)
        values1 = parents[sample1, range(size)]
        values2 = parents[sample2, range(size)]
        values = (values1 + values2) / 2
        newFitness = fitness(values)
        improved = False
        if newFitness <= worstFitness:
            sd1 = parentsSD[sample1, range(size)]
            sd2 = parentsSD[sample2, range(size)]
            sd = (sd1 + sd2) / 2
            
            parentsFitness[worstSolution] = newFitness
            parents[worstSolution,:] = values
            parentsSD[worstSolution,:] = sd
            
            improved = True
            worstSolution = np.argmax(parentsFitness)
            worstFitness = parentsFitness[worstSolution]

        observer.update(iter, parentsFitness, improved)

    bestSolution = np.argmin(parentsFitness)
    bestFitness = parentsFitness[bestSolution]
    results = {
        "fitness": bestFitness,
        "solution": parents[bestSolution],
        "observer": observer
    }

    return results
