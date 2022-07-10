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

def test_findNondominatedSolutions():
    fitnessValues1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    result1 = m.findNondominatedSolutions(fitnessValues1)
    assert result1 == {0, 1, 2}

    fitnessValues2 = np.array([
        [5, 1, 0],
        [4, 1, 0],
        [0, 0, 1],
        [-3, 0, 2],
    ])
    result2 = m.findNondominatedSolutions(fitnessValues2)
    assert result2 == {1, 2, 3}

    fitnessValues3 = np.array([
        [0.5, -1, 2.3, 4],
        [1.2, -1, 2.7, 4]
    ])
    result3 = m.findNondominatedSolutions(fitnessValues3)
    assert result3 == {0}

def test_nonDominatedSort():
    fitnessValues1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    result1 = m.nonDominatedSort(fitnessValues1)
    assert np.array_equal(result1, [1, 1, 1])

    fitnessValues2 = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [3, 2, 1],
    ])
    result2 = m.nonDominatedSort(fitnessValues2)
    assert np.array_equal(result2, [2, 1, 2, 2, 3])

    fitnessValues3 = np.array([
        [1, 0, 0]
    ])
    result3 = m.nonDominatedSort(fitnessValues3)
    assert np.array_equal(result3, [1])

def test_crowdingDistanceAssignment():
    fitnessValues1 = np.array([
        [1, 0],
        [0, 0],
    ])
    result1 = m.crowdingDistanceAssignment(fitnessValues1)
    assert np.array_equal(result1, [np.Inf, np.Inf])

    fitnessValues2 = np.array([
        [6, 1],
        [0, 0],
        [-4, -1],
    ])
    result2 = m.crowdingDistanceAssignment(fitnessValues2)
    assert np.array_equal(result2, [np.Inf, 2, np.Inf])

    fitnessValues3 = np.array([
        [6, 1],
        [0, 0],
        [1, 0.5],
        [-4, -1],
    ])
    result3 = m.crowdingDistanceAssignment(fitnessValues3)
    assert np.array_equal(result3, [np.Inf, 1.25, 1.1, np.Inf])

def test_vega():
    np.random.seed(0)
    fitnessFunctions = [
        lambda ch: (ch.values[0] - 1) ** 2,
        lambda ch: (ch.values[0] + 1) ** 2
    ]
    populationSize = 100
    iterations = 1000
    chromosomeFactory = chr.NumericChromosomeFactory(1, -5000, 5000)
    crossover = cr.SimulatedBinaryCrossover()
    mutation = mu.NormalDistributionMutation(1)
    
    
    for _ in range(3):
        results = m.vega(fitnessFunctions, chromosomeFactory, populationSize,
                            crossover, mutation, iterations)
        assert results["fitnessValues"].shape[0] >= 2
        assert results["fitnessValues"].shape[1] == 2
        assert np.all(results["fitnessValues"] < 5)
        assert results["fitnessValues"].shape[0] == len(results["solutions"])

def test_vegaOddPopulation():
    np.random.seed(0)
    fitnessFunctions = [
        lambda ch: (ch.values[0] - 1) ** 2,
        lambda ch: (ch.values[0] + 1) ** 2
    ]
    populationSize = 101
    iterations = 1000
    chromosomeFactory = chr.NumericChromosomeFactory(1, -5000, 5000)
    crossover = cr.SimulatedBinaryCrossover()
    mutation = mu.NormalDistributionMutation(1)
    
    
    for _ in range(3):
        results = m.vega(fitnessFunctions, chromosomeFactory, populationSize,
                            crossover, mutation, iterations)
        assert results["fitnessValues"].shape[0] >= 2
        assert results["fitnessValues"].shape[1] == 2
        assert np.all(results["fitnessValues"] < 5)
        assert results["fitnessValues"].shape[0] == len(results["solutions"])

def test_nsga2():
    np.random.seed(0)
    fitnessFunctions = [
        lambda ch: (ch.values[0] - 490) ** 2,
        lambda ch: (ch.values[0] - 510) ** 2
    ]
    populationSize = 40
    iterations = 200
    chromosomeFactory = chr.NumericChromosomeFactory(1, -5000, 5000)
    crossover = cr.SimulatedBinaryCrossover()
    mutation = mu.NormalDistributionMutation(1)

    for _ in range(3):
        results = m.nsga2(fitnessFunctions, chromosomeFactory, populationSize,
                            crossover, mutation, iterations)
        assert results["fitnessValues"].shape[0] >= 2
        assert results["fitnessValues"].shape[1] == 2
        assert np.all(results["fitnessValues"] < 500)
        assert results["fitnessValues"].shape[0] == len(results["solutions"])

def test_nsga2OddPopulation():
    np.random.seed(0)
    fitnessFunctions = [
        lambda ch: (ch.values[0] - 490) ** 2,
        lambda ch: (ch.values[0] - 510) ** 2
    ]
    populationSize = 41
    iterations = 200
    chromosomeFactory = chr.NumericChromosomeFactory(1, -5000, 5000)
    crossover = cr.SimulatedBinaryCrossover()
    mutation = mu.NormalDistributionMutation(1)

    for _ in range(3):
        results = m.nsga2(fitnessFunctions, chromosomeFactory, populationSize,
                            crossover, mutation, iterations)
        assert results["fitnessValues"].shape[0] >= 2
        assert results["fitnessValues"].shape[1] == 2
        assert np.all(results["fitnessValues"] < 500)
        assert results["fitnessValues"].shape[0] == len(results["solutions"])
