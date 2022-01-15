import numpy as np

class WeightedSumFitness:
    def __init__(self, weights, fitnessFunctions):
        self.weights = np.copy(weights)
        self.fitnessFunctions = list(fitnessFunctions)

    def __call__(self, *args, **kwds):
        fitnessValues = [f(args[0]) for f in self.fitnessFunctions]
        return np.sum(self.weights * fitnessValues)

class MultiobjectiveObserver:
    def __init__(self, iterations, numberOfFitness):
        self.minFitness = np.zeros((iterations + 1, numberOfFitness))
        self.meanFitness = np.zeros((iterations + 1, numberOfFitness))
        self.maxFitness = np.zeros((iterations + 1, numberOfFitness))

    def update(self, iteration, fitnessValues, population):
        self.minFitness[iteration,:] = np.apply_along_axis(np.min, 0, fitnessValues)
        self.meanFitness[iteration,:] = np.apply_along_axis(np.mean, 0, fitnessValues)
        self.maxFitness[iteration,:] = np.apply_along_axis(np.max, 0, fitnessValues)

def findNondominatedSolutions(fitnessValues):
    size = fitnessValues.shape[0]
    nondominatedSet = set(range(size))
    for i in range(size):
        better = fitnessValues > fitnessValues[i,]
        better = np.any(better, axis = 1)
        notWorse = fitnessValues >= fitnessValues[i,]
        notWorse = np.all(notWorse, axis = 1)
        dominated = np.logical_and(better, notWorse)
        dominated = np.flatnonzero(dominated)
        print(better)
        print(notWorse)
        print(dominated)
        nondominatedSet.difference_update(set(dominated))

    return nondominatedSet

def vega(fitnessFunctions, chromosomeFactory, populationSize,
        crossover, mutation, iterations):
    numberOfFitness = len(fitnessFunctions)
    m = np.ceil(populationSize / numberOfFitness)
    observer = MultiobjectiveObserver(iterations, numberOfFitness)
    population = [chromosomeFactory.createChromosome() for _ in range(populationSize)]
    populationNew = [chromosomeFactory.createChromosome() for _ in range(m * numberOfFitness)]
    if populationSize % 2 == 0:
        isOdd = False
    else:
        isOdd = True
        dummyChromosome = chromosomeFactory.createChromosome()
        dummyChromosome.initialize()
    
    for i in range(populationSize):
        population[i].initialize()

    for iter in range(iterations):
        fitnessValues = np.array([[f(s) for f in fitnessFunctions] for s in population])
        observer.update(iter, fitnessValues, population)
        for i in numberOfFitness:
            tournamentIndex1 = np.random.randint(0, populationSize, m)
            tournamentIndex2 = np.random.randint(0, populationSize, m)
            for j in range(m):
                if fitnessValues[tournamentIndex1[j], i] < fitnessValues[tournamentIndex2[j], i]:
                    populationNew[i * m + j].copyValues(population[tournamentIndex1[j]])
                else:
                    populationNew[i * m + j].copyValues(population[tournamentIndex2[j]])
        
        parentIndex1 = np.random.randint(0, m * numberOfFitness, populationSize // 2)
        parentIndex2 = np.random.randint(0, m * numberOfFitness, populationSize // 2)
        for i in range(populationSize // 2):
            crossover.crossover(populationNew[parentIndex1[i]], populationNew[parentIndex2[i]],
                population[2 * i], population[2 * i + 1])
        if isOdd == True:
            parent1 = np.random.randint(0, m * numberOfFitness)
            parent2 = np.random.randint(0, m * numberOfFitness)
            crossover.crossover(populationNew[parent1], populationNew[parent2],
                population[populationSize // 2 + 1], dummyChromosome)

        for i in range(populationSize):
            mutation.mutation(population[i])

    fitnessValues = np.array([[f(s) for f in fitnessFunctions] for s in population])
    observer.update(iter, fitnessValues, population)
    nondominated = findNondominatedSolutions(fitnessValues)

    results = {
        "fitnessValues": fitnessValues,
        "solutions": [population[i] for i in nondominated],
        "observer": observer
    }

    return results
