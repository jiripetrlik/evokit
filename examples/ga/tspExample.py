import numpy as np
import evokit.chromosome as chr
import evokit.crossover as cr
import evokit.mutation as mu
import evokit.ga as ga

distanceMatrix = np.array([
    [0, 2, 5, 7, 5],
    [2, 0, 4, 8, 1],
    [5, 4, 0, 9, 2],
    [7, 8, 9, 0, 9],
    [5, 1, 2, 9, 0]
])

def fitnessFunction(path):
    length = 0
    previousPoint = path.values[-1]
    for point in path.values:
        length += distanceMatrix[previousPoint, point]

    return length

def main():
    chromosomeFactory = chr.PermutationChromosomeFactory(distanceMatrix.shape[0])
    crossover = cr.PartiallyMappedCrossover()
    mutation = mu.ReciprocalExchangeMutation()

    result = ga.geneticAlgorithm(fitnessFunction, chromosomeFactory, 20,
                crossover, mutation, 100)
    
    print("Best fitness:", result["fitness"])
    print("Best solution:", result["solution"].values)
    result["observer"].plot()

if __name__ == "__main__":
    main()
