import numpy as np
import evokit.chromosome as ch

def test_initialize_numeric_chromosome_1():
    np.random.seed(0)

    chromosome = ch.NumericChromosome(5, 1, 7)
    chromosome.initialize()
    for _ in range(100):
        for i in range(len(chromosome.values)):
            assert chromosome.values[i] >= 1
            assert chromosome.values[i] <= 7

def test_initialize_numeric_chromosome_2():
    np.random.seed(0)

    min = np.array([-1, 2.5, 4])
    max = np.array([-0.5, 9, 150])
    chromosome = ch.NumericChromosome(3, min, max)
    chromosome.initialize()
    for _ in range(100):
        for i in range(len(chromosome.values)):
            assert chromosome.values[i] >= min[i]
            assert chromosome.values[i] <= max[i]

def test_initialize_numeric_chromosome_3():
    np.random.seed(0)

    min = -5
    max = np.array([-0.5, 9, 150])
    chromosome = ch.NumericChromosome(3, min, max)
    chromosome.initialize()
    for _ in range(100):
        for i in range(len(chromosome.values)):
            assert chromosome.values[i] >= min
            assert chromosome.values[i] <= max[i]
