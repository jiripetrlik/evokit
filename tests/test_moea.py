import numpy as np
import evokit.moea as m
import evokit.chromosome as chr
import evokit.crossover as cr
import evokit.mutation as mu
import evokit.ga as ga

def test_weightedFitness():
    weights = [2, 7, 11]
    fitnessFunctions = [
        lambda x: x[0] ** 2 + 5,
        lambda x: x[0] - 7,
        lambda x: 2 * x[0] - 1
    ]
    weightedFitnessFunction = m.WeightedSumFitness(weights, fitnessFunctions)

    assert weightedFitnessFunction([2]) == 16
    assert weightedFitnessFunction([5]) == 145
    assert weightedFitnessFunction([-2]) == -100

def test_weightedSumGA():
    np.random.seed(0)
    fitnessFunctions = [
        lambda x: x.values[0] ** 2,
        lambda x: (x.values[0] - 1) ** 2
    ]
    weights = [
        [0, 1],
        [1, 0],
        [0.5, 0.5]
    ]
    expectedResults = [1, 0, 0.5]
    expectedFitness = [0, 0, 0.25]

    populationSize = 100
    iterations = 1000
    for i in range(len(weights)):        
        weightedSumFitness = m.WeightedSumFitness(weights[i], fitnessFunctions)
        chromosomeFactory = chr.NumericChromosomeFactory(1, -5000, 5000)
        crossover = cr.SimulatedBinaryCrossover()
        mutation = mu.NormalDistributionMutation(1)

        result = ga.geneticAlgorithm(weightedSumFitness, chromosomeFactory, populationSize,
                    crossover, mutation, iterations)
        
        assert abs(result["fitness"] - expectedFitness[i]) < 0.01
        assert abs(result["solution"].values[0] - expectedResults[i]) < 0.01
