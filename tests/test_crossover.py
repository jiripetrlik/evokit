import numpy as np
import evokit.chromosome as ch
import evokit.crossover as cr

def test_onePointCrossover():
    np.random.seed(0)
    size = 128
    crossover = cr.OnePointCrossover()
    for _ in range(100):
        parent1 = ch.BinaryChromosome(size)
        parent1.initialize()
        parent2 = ch.BinaryChromosome(size)
        parent2.initialize()
        child1 = ch.BinaryChromosome(size)
        child1.initialize()
        child2 = ch.BinaryChromosome(size)
        child2.initialize()

        sumParents = np.sum(parent1.values) + np.sum(parent2.values)
        crossover.crossover(parent1, parent2, child1, child2)
        sumChildren = np.sum(child1.values) + np.sum(child2.values)
        assert sumParents == sumChildren

def test_uniformCrossover():
    np.random.seed(0)
    size = 128
    crossover = cr.UniformCrossover()
    for _ in range(100):
        parent1 = ch.BinaryChromosome(size)
        parent1.initialize()
        parent2 = ch.BinaryChromosome(size)
        parent2.initialize()
        child1 = ch.BinaryChromosome(size)
        child1.initialize()
        child2 = ch.BinaryChromosome(size)
        child2.initialize()

        sumParents = np.sum(parent1.values) + np.sum(parent2.values)
        crossover.crossover(parent1, parent2, child1, child2)
        sumChildren = np.sum(child1.values) + np.sum(child2.values)
        assert sumParents == sumChildren

def test_simulatedBinaryCrossover():
    np.random.seed(0)
    size = 3
    min = np.array([0, 1, 2])
    max = np.array([5, 6, 7])

    parent1 = ch.NumericChromosome(size, min, max)
    parent1.initialize()
    parent2 = ch.NumericChromosome(size, min, max)
    parent2.initialize()
    child1 = ch.NumericChromosome(size, min, max)
    child1.initialize()
    child2 = ch.NumericChromosome(size, min, max)
    child2.initialize()

    crossover = cr.SimulatedBinaryCrossover()

    for _ in range(100):
        crossover.crossover(parent1, parent2, child1, child2)
        for i in range(size):
            assert child1.values[i] >= min[i]
            assert child1.values[i] <= max[i]

            assert child2.values[i] >= min[i]
            assert child2.values[i] <= max[i]
