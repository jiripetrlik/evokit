import numpy as np
import matplotlib.pyplot as plt

class PsoObserver:
    def __init__(self, iterations):
        self.minFitness = np.zeros(iterations)
        self.meanFitness = np.zeros(iterations)
        self.maxFitness = np.zeros(iterations)

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

def pso(fitnessFunction, size, minValues, maxValues, populationSize, iterations):
    observer = PsoObserver(iterations)
    solutions = np.random.uniform(low = minValues,
                                    high = maxValues,
                                    size = (populationSize, size))
    velocityInitialRange = 0.1 * (np.array(maxValues) - np.array(minValues))
    velocity = np.random.uniform(low = -velocityInitialRange / 2,
                                    high = velocityInitialRange / 2,
                                    size = (populationSize, size))
    fitnessValues = np.apply_along_axis(fitnessFunction, 1, solutions)
    bestPosition = np.copy(solutions)
    bestFitnessValues = np.copy(fitnessValues)

    phiMax1 = 1.108
    phiMax2 = 1.108
    w = 0.72
    for iter in range(iterations):
        bestNeighbor = np.argmin(fitnessValues)
        phi1 = np.random.uniform(low = 0, high = phiMax1, size = (populationSize, size))
        phi2 = np.random.uniform(low = 0, high = phiMax2, size = (populationSize, size))
        velocity = (w * velocity +
                    phi1 * (bestPosition - solutions) +
                    phi2 * (solutions[bestNeighbor] - solutions))
        solutions += velocity

        fitnessValues = np.apply_along_axis(fitnessFunction, 1, solutions)
        updateBestPosition = fitnessValues <= bestFitnessValues
        bestFitnessValues[updateBestPosition] = fitnessValues[updateBestPosition]
        bestPosition[updateBestPosition,:] = solutions[updateBestPosition,:]

        observer.update(iter, fitnessValues, solutions)

    bestSolution = np.argmin(fitnessValues)
    results = {
        "fitness": fitnessValues[bestSolution],
        "solution": solutions[bestSolution],
        "observer": observer
    }

    return results
