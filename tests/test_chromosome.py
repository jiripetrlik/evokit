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

def test_renormalize():
    min = -5
    max = np.array([-0.5, 9, 150])
    chromosome = ch.NumericChromosome(3, min, max)
    chromosome.values = np.array([-7, 10, 151])
    chromosome.renormalize()
    assert np.array_equal(chromosome.values, [-5, 9, 150])

def test_copyValues():
    np.random.seed(0)

    chromosome1 = ch.NumericChromosome(5, -1, 2)
    chromosome1.initialize()
    chromosome2 = ch.NumericChromosome(5, -1, 2)
    chromosome2.initialize()
    assert chromosome1.values is not chromosome2.values
    assert np.array_equal(chromosome1.values, chromosome2.values) == False
    
    chromosome1.copyValues(chromosome2)
    assert chromosome1.values is not chromosome2.values
    assert np.array_equal(chromosome1.values, chromosome2.values)
