import numpy as np
import evokit.chromosome as chr
import evokit.crossover as cr
import evokit.mutation as mu
import evokit.ga as ga

def test_geneticAlgorithm():
    np.random.seed(0)
    fitness = lambda ch : ((ch.values[0] - 1) ** 2) + (ch.values[1] ** 2) + ((ch.values[2] + 1) ** 2)
    populationSize = 100
    chromosomeFactory = chr.NumericChromosomeFactory(3, -5000, 5000)
    crossover = cr.SimulatedBinaryCrossover()
    mutation = mu.NormalDistributionMutation(1)
    iterations = 1000

    for _ in range(3):
        result = ga.geneticAlgorithm(fitness, chromosomeFactory, populationSize,
                    crossover, mutation, iterations)
        assert result["fitness"] < 0.01
        assert np.abs(result["solution"].values[0] - 1) < 0.1
        assert np.abs(result["solution"].values[1]) < 0.1
        assert np.abs(result["solution"].values[2] + 1) < 0.1
        assert len(result["observer"].minFitness) == 1001
        assert len(result["observer"].meanFitness) == 1001
        assert len(result["observer"].maxFitness) == 1001
