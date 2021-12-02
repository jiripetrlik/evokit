import numpy as np
import matplotlib.pyplot as plt

MAX_MOVING_WINDOW_SIZE = 30
SELF_ADAPTATION_C = 0.817
SELF_ADAPTATION_RATE = 0.2

class EvolutionStrategyObserver:
    def __init__(self, iterations):
        self.minFitness = np.zeros(iterations)
        self.meanFitness = np.zeros(iterations)
        self.maxFitness = np.zeros(iterations)
        self.previousParents = None
        self.improved = np.zeros(iterations, dtype=np.bool_)

    def update(self, iteration, parentsFitness, parents):
        self.minFitness[iteration] = np.min(parentsFitness)
        self.meanFitness[iteration] = np.mean(parentsFitness)
        self.maxFitness[iteration] = np.max(parentsFitness)
        if np.array_equal(self.previousParents, parents):
            self.improved[iteration] = True

    def plot(self):
        plt.plot(self.minFitness, label = "Min. fitness")
        plt.plot(self.meanFitness, label = "Mean fitness")
        plt.plot(self.maxFitness, label = "Max. fitness")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()

def onePlusOneES(fitness, size, sd, iterations, adaptive = True):
    vector1 = np.random.normal(scale = sd, size = size)
    bestFintess = fitness(vector1)
    observer = EvolutionStrategyObserver(iterations)
    if adaptive == True:
        numberOfImprovements = 0
        if size < MAX_MOVING_WINDOW_SIZE:
            movingWindow = size
        else:
            movingWindow = MAX_MOVING_WINDOW_SIZE

    for iter in range(iterations):
        vector2 = vector1 + np.random.normal(scale = sd, size = size)
        newFitness = fitness(vector2)
        if newFitness <= bestFintess:
            bestFintess = newFitness
            vector1 = vector2
            if adaptive == True:
                numberOfImprovements += 1
        if adaptive == True and (iter + 1) % movingWindow == 0:
            if numberOfImprovements / movingWindow > SELF_ADAPTATION_RATE:
                sd = sd / SELF_ADAPTATION_C
            else:
                sd = sd * SELF_ADAPTATION_C
            numberOfImprovements = 0

        observer.update(iter, [bestFintess], vector1)
        
    results = {
        "fitness": bestFintess,
        "solution": vector1,
        "observer": observer
    }

    return results

def miPlusOneES(fitness, size, mi, iterations):
    parents = np.random.normal(size=(mi, size))
    parentsSD = np.random.uniform(low = 0, high = 1, size=(mi, size))
    parentsFitness = np.apply_along_axis(fitness, 1, parents)
    worstSolution = np.argmax(parentsFitness)
    worstFitness = parentsFitness[worstSolution]
    observer = EvolutionStrategyObserver(iterations)

    for iter in range(iterations):
        sample1 = np.random.randint(low = 0, high = mi, size = size)
        sample2 = np.random.randint(low = 0, high = mi, size = size)
        values1 = parents[sample1, range(size)]
        values2 = parents[sample2, range(size)]
        values = (values1 + values2) / 2
        sd1 = parentsSD[sample1, range(size)]
        sd2 = parentsSD[sample2, range(size)]
        sd = (sd1 + sd2) / 2
        values = values + np.random.normal(scale=sd)

        newFitness = fitness(values)
        improved = False
        if newFitness <= worstFitness:
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

def miPlusLambdaES(fitness, size, mi, l, iterations):
    population = np.random.normal(size=(mi + l, size))
    populationSD = np.random.uniform(low = 0, high = 1, size=(mi + l, size))
    fitnessValues = np.zeros(shape=(mi + l))
    fitnessValues[:mi] = np.apply_along_axis(fitness, 1, population[:mi,:])
    observer = EvolutionStrategyObserver(iterations)

    for iter in range(iterations):
        for i in range(l):
            sample1 = np.random.randint(low = 0, high = mi, size = size)
            sample2 = np.random.randint(low = 0, high = mi, size = size)
            values1 = population[sample1, range(size)]
            values2 = population[sample2, range(size)]
            values = (values1 + values2) / 2
            sd1 = populationSD[sample1, range(size)]
            sd2 = populationSD[sample2, range(size)]
            sd = (sd1 + sd2) / 2
            values = values + np.random.normal(scale=sd)            

            population[mi + i,:] = values
            populationSD[mi + i,:] = sd

        fitnessValues[mi:] = np.apply_along_axis(fitness, 1, population[mi:,:])
        order = np.argsort(fitnessValues)
        
        fitnessValues = fitnessValues[order]
        population = population[order,:]
        populationSD = populationSD[order,:]
        observer.update(iter, fitnessValues[:mi], False)

    results = {
        "fitness": fitnessValues[0],
        "solution": population[0,:],
        "observer": observer
    }

    return results
