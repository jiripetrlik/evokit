import numpy as np
import evokit.chromosome as ch

def test_initialize_numeric_chromosome():
    np.random.seed(0)

    chromosome = ch.NumericChromosome(5, 1, 7)
    chromosome.initialize()
    for _ in range(100):
        for i in range(len(chromosome.values)):
            assert chromosome.values[i] >= 1
            assert chromosome.values[i] <= 7
