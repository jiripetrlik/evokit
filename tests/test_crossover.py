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

def test_partiallyMappedCrossover():
    np.random.seed(0)
    size = 256

    changed1 = 0
    changed2 = 0
    for _ in range(100):
        parent1 = ch.PermutationChromosome(size)
        parent2 = ch.PermutationChromosome(size)
        child1 = ch.PermutationChromosome(size)
        child1OriginalValues = np.copy(child1.values)
        child2 = ch.PermutationChromosome(size)
        child2OriginalValues = np.copy(child2.values)

        crossover = cr.PartiallyMappedCrossover()
        crossover.crossover(parent1, parent2, child1, child2)

        assert not np.array_equal(child1OriginalValues, child1.values)
        assert not np.array_equal(child2OriginalValues, child2.values)
        assert len(np.unique(child1.values)) == size
        assert len(np.unique(child2.values)) == size
        if not np.array_equal(child1.values, parent1.values):
            changed1 += 1
        if not np.array_equal(child2.values, parent2.values):
            changed2 += 1

    assert changed1 > 0
    assert changed2 > 0
