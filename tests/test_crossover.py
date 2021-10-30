import numpy as np
import evokit.chromosome as ch
import evokit.crossover as cr

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
