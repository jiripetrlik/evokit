import numpy as np
import matplotlib.pyplot as plt

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
        nondominatedSet.difference_update(set(dominated))

    return nondominatedSet

def plotNondominatedSolutions(fitnessValues, firstObjective = 0, secondObjective = 1):
    nondominated = findNondominatedSolutions(fitnessValues)
    nonDominatedFitnessValues = fitnessValues[tuple(nondominated),:]
    
    plt.scatter(nonDominatedFitnessValues[:,firstObjective],
                nonDominatedFitnessValues[:,secondObjective])
    plt.show()

def nonDominatedSort(fitnessValues):
    size = fitnessValues.shape[0]
    s = [set() for _ in range(size)]
    n = np.zeros(size)
    rank = np.zeros(size)
    f = set()

    for p in range(size):
        for q in range(size):
            better = fitnessValues[p,] < fitnessValues[q,]
            better = np.any(better)
            notWorse = fitnessValues[p,] <= fitnessValues[q,]
            notWorse = np.all(notWorse)
            if np.logical_and(better, notWorse):
                s[p].add(q)

            better = fitnessValues[p,] > fitnessValues[q,]
            better = np.any(better)
            notWorse = fitnessValues[p,] >= fitnessValues[q,]
            notWorse = np.all(notWorse)
            if np.logical_and(better, notWorse):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 1
            f.add(p)

    i = 1
    while len(f) > 0:
        fNext = set()
        for p in f:
            for q in s[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    fNext.add(q)
        
        i += 1
        f = fNext

    return rank

def vega(fitnessFunctions, chromosomeFactory, populationSize,
        crossover, mutation, iterations):
    numberOfFitness = len(fitnessFunctions)
    m = int(np.ceil(populationSize / numberOfFitness))
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
        for i in range(numberOfFitness):
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
        "fitnessValues": fitnessValues[tuple(nondominated),:],
        "solutions": [population[i].values for i in nondominated],
        "observer": observer
    }

    return results

def crowdingDistanceAssignment(fitnessValues):
    populationSize = fitnessValues.shape[0]
    crowdingDistance = np.zeros(populationSize)
    if populationSize < 2:
        crowdingDistance += np.Inf
        return crowdingDistance
    numberOfFitness = fitnessValues.shape[1]
    for i in range(numberOfFitness):
        cDistance = np.zeros(populationSize)
        fOrder = np.argsort(fitnessValues[:,i])
        sortedValues = fitnessValues[tuple(fOrder),i]
        fRange = sortedValues[populationSize - 1] - sortedValues[0]
        if fRange == 0:
            fRange = 1
        cDistance[0] = np.Inf
        cDistance[populationSize - 1] = np.Inf
        leftPart = sortedValues[:populationSize - 2]
        rightPart = sortedValues[2:]
        cDistance[1:populationSize - 1] += (rightPart - leftPart) / fRange

        crowdingDistance += cDistance
    
    return crowdingDistance

def nsga2(fitnessFunctions, chromosomeFactory, populationSize,
            crossover, mutation, iterations):
    numberOfFitness = len(fitnessFunctions)
    observer = MultiobjectiveObserver(iterations, numberOfFitness)
    population = [chromosomeFactory.createChromosome() for _ in range(2 * populationSize)]
    for i in range(2 * populationSize):
        population[i].initialize()
    if populationSize % 2 == 0:
        isOdd = False
    else:
        isOdd = True
        dummyChromosome = chromosomeFactory.createChromosome()
        dummyChromosome.initialize()
    newPopulationTuple = tuple([i + populationSize for i in range(populationSize)])
    fitnessValues = np.array([[f(s) for f in fitnessFunctions] for s in population])
    rank = nonDominatedSort(fitnessValues)
    order = np.argsort(rank)
    population = [population[index] for index in order]

    for iter in range(iterations):
        parentIndex1 = np.random.randint(populationSize, size=[2, populationSize // 2])
        parentIndex1 = np.min(parentIndex1, axis = 0)
        parentIndex2 = np.random.randint(populationSize, size=[2, populationSize // 2])
        parentIndex2 = np.min(parentIndex2, axis = 0)
        for i in range(populationSize // 2):
            crossover.crossover(population[parentIndex1[i]], population[parentIndex2[i]],
                                population[populationSize + 2 * i],
                                population[populationSize + 2 * i + 1])
        if isOdd == True:
            oddParent1 = np.min(np.random.randint(populationSize))
            oddParent2 = np.min(np.random.randint(populationSize))
            crossover.crossover(population[oddParent1],
                                population[oddParent2],
                                population[2 * populationSize - 1],
                                dummyChromosome)
        for i in range(populationSize):
            mutation.mutation(population[i + populationSize])
        newFitnessValues = [[f(population[i]) for f in fitnessFunctions]
                                for i in range(populationSize)]
        fitnessValues[newPopulationTuple,] = newFitnessValues
        rank = nonDominatedSort(fitnessValues)
        order = np.argsort(rank)
        population = [population[i] for i in order]
        fitnessValues = fitnessValues[tuple(order),]
        rank = rank[order]
        if rank[populationSize - 1] == rank[populationSize]:
            subpopulation = tuple(np.where(rank == rank[populationSize - 1])[0].tolist())
            subpopulationFitness = fitnessValues[subpopulation,:]
            crowdingDistance = crowdingDistanceAssignment(subpopulationFitness)
            cdOrder = tuple(np.flip(np.argsort(crowdingDistance)).tolist())
            fitnessValues[subpopulation,:] = subpopulationFitness[cdOrder,:]

        observer.update(iter, fitnessValues[0:populationSize,:], population[0:populationSize])

    fitnessValues = fitnessValues[0:populationSize,:]
    population = population[0:populationSize]
    nondominated = findNondominatedSolutions(fitnessValues)

    results = {
        "fitnessValues": fitnessValues[tuple(nondominated),:],
        "solutions": [population[i].values for i in nondominated],
        "observer": observer
    }

    return results    
