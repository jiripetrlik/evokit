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

    def plot(self):
        for i in range(self.minFitness.shape[1]):
            text = "Min fitness " + str(i + 1)
            plt.plot(self.minFitness[:,i], label = text)
        
        plt.legend()
        plt.show()

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
    n = np.zeros(size)
    rank = np.zeros(size)

    a = np.repeat(fitnessValues, size, axis=0)
    b = np.tile(fitnessValues, (size, 1))
    isBetterArray = a < b
    isNotWorseArray = a <= b
    isBetter = np.apply_along_axis(np.any, 1, isBetterArray)
    isNotWorse = np.apply_along_axis(np.all, 1, isNotWorseArray)
    dominates = np.logical_and(isBetter, isNotWorse)
    dominatedSets = [np.where(dominates[i * size : (i + 1) * size])[0] for i in range(size)]
    s = [set(dSet) for dSet in dominatedSets]
    for dSet in dominatedSets:
        n[dSet] += 1
    f = np.where(n == 0)[0]
    rank[f] = 1

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

        crowdingDistance[fOrder] += cDistance
    
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

    # f = open("debug.txt", "w")
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
        newFitnessValues = [[f(population[populationSize + i]) for f in fitnessFunctions]
                                for i in range(populationSize)]
        fitnessValues[newPopulationTuple,] = newFitnessValues
        rank = nonDominatedSort(fitnessValues)
        order = np.argsort(rank)
        population = [population[i] for i in order]
        fitnessValues = fitnessValues[tuple(order),]
        rank = rank[order]
        # crowdingDistanceDebug = np.empty(2 * populationSize)
        # crowdingDistanceDebug[:] = np.nan
        if rank[populationSize - 1] == rank[populationSize]:
            subpopulationIndex = tuple(np.where(rank == rank[populationSize - 1])[0].tolist())
            subpopulationFitness = fitnessValues[subpopulationIndex,:]
            crowdingDistance = crowdingDistanceAssignment(subpopulationFitness)
            cdOrder = tuple(np.flip(np.argsort(crowdingDistance)).tolist())
            fitnessValues[subpopulationIndex,:] = subpopulationFitness[cdOrder,:]
            subpopulation = [population[subpopulationIndex[cdOrder[i]]]
                                for i in range(len(subpopulationIndex))]
            # crowdingDistanceDebug[np.array(subpopulationIndex)] = [crowdingDistance[cdOrder[i]]
            #                             for i in range(len(subpopulation))]
            for i in range(len(subpopulation)):
                population[subpopulationIndex[i]] = subpopulation[i]
            
#        for i in range(2 * populationSize):
#            debugString = str(iter) + "," + str(i)
#            debugString += "," + str(population[i].values[0])
#            for j in range(numberOfFitness):
#                debugString += "," + str(fitnessValues[i,j])
#            
#            debugString += "," + str(rank[i])
#            debugString += "," + str(crowdingDistanceDebug[i])
#            debugString += "\n"

#            f.write(debugString)

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
