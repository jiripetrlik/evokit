import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithmObserver:
    def __init__(self, iterations):
        self.minFitness = np.zeros(iterations + 1)
        self.meanFitness = np.zeros(iterations + 1)
        self.maxFitness = np.zeros(iterations + 1)

    def update(self, iteration, fitnessValues, population):
        self.minFitness[iteration] = np.min(fitnessValues)
        self.meanFitness[iteration] = np.mean(fitnessValues)
        self.maxFitness[iteration] = np.max(fitnessValues)

    def plot(self):
        plt.plot(self.minFitness, label = "Min. fitness")
        plt.plot(self.meanFitness, label = "Mean fitness")
        plt.plot(self.maxFitness, label = "Max. fitness")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()

def geneticAlgorithm(fitnessFunction, chromosomeFactory, populationSize,
        crossover, mutation, iterations, elitism = True):
    observer = GeneticAlgorithmObserver(iterations)
    population = [chromosomeFactory.createChromosome() for _ in range(populationSize)]
    populationNew = [chromosomeFactory.createChromosome() for _ in range(populationSize)]
    bestSolution = chromosomeFactory.createChromosome()
    if populationSize % 2 == 0:
        isOdd = False
    else:
        isOdd = True
        dummyChromosome = chromosomeFactory.createChromosome()
        dummyChromosome.initialize()

    for i in range(populationSize):
        population[i].initialize()

    for iter in range(iterations):
        fitnessValues = [fitnessFunction(chromosome) for chromosome in population]
        if elitism == True:
            bestSolutionIndex = np.argmin(fitnessValues)
            bestSolution.copyValues(population[bestSolutionIndex])
        observer.update(iter, fitnessValues, population)

        tournamentIndex1 = np.random.randint(0, populationSize, populationSize)
        tournamentIndex2 = np.random.randint(0, populationSize, populationSize)
        for i in range(populationSize):
            if fitnessValues[tournamentIndex1[i]] < fitnessValues[tournamentIndex2[i]]:
                populationNew[i].copyValues(population[tournamentIndex1[i]])
            else:
                populationNew[i].copyValues(population[tournamentIndex2[i]])

        parentIndex1 = np.random.randint(0, populationSize, populationSize // 2)
        parentIndex2 = np.random.randint(0, populationSize, populationSize // 2)
        for i in range(populationSize // 2):
            crossover.crossover(populationNew[parentIndex1[i]], populationNew[parentIndex2[i]],
                population[2 * i], population[2 * i + 1])
        if isOdd == True:
            parent1 = np.random.randint(0, populationSize)
            parent2 = np.random.randint(0, populationSize)
            crossover.crossover(populationNew[parent1], populationNew[parent2],
                population[populationSize // 2 + 1], dummyChromosome)

        for i in range(populationSize):
            mutation.mutation(population[i])
        
        if elitism == True:
            population[0].copyValues(bestSolution)

    fitnessValues = [fitnessFunction(chromosome) for chromosome in population]
    observer.update(iterations, fitnessValues, population)

    bestSolution = np.argmin(fitnessValues)
    results = {
        "fitness": fitnessValues[bestSolution],
        "solution": population[bestSolution],
        "observer": observer
    }

    return results
