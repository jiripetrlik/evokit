import numpy as np
import evokit.chromosome as ch

def test_binary_chromosome_factory():
    size = 32
    factory = ch.BinaryChromosomeFactory(size)
    chromosome = factory.createChromosome()

    assert len(chromosome.values) == size

def test_initialize_binary_chromosome():
    np.random.seed(0)

    size = 128
    for _ in range(100):
        chromosome = ch.BinaryChromosome(size)
        chromosome.initialize()

        assert chromosome.values.dtype == np.dtype(bool)
        assert len(chromosome.values) == size
        assert np.sum(chromosome.values) > 0

def test_copyValues_binary_chromosome():
    np.random.seed(0)

    size = 128
    chromosome1 = ch.BinaryChromosome(size)
    chromosome1.initialize()
    chromosome2 = ch.BinaryChromosome(size)
    chromosome2.initialize()
    assert chromosome1.values is not chromosome2.values
    assert np.array_equal(chromosome1.values, chromosome2.values) == False

    chromosome1.copyValues(chromosome2)
    assert chromosome1.values is not chromosome2.values
    assert np.array_equal(chromosome1.values, chromosome2.values)

def test_getInteger_binary_chromosome():
    size = 24
    chromosome = ch.BinaryChromosome(size)
    chromosome.values = np.zeros(size, dtype=np.dtype(bool))
    assert chromosome.getInteger(8, 16) == 0

    chromosome.values = np.zeros(size, dtype=np.dtype(bool))
    chromosome.values[9] =True
    chromosome.values[10] =True
    assert chromosome.getInteger(8, 16) == 6

    chromosome.values = np.ones(size, dtype=np.dtype(bool))
    assert chromosome.getInteger(8, 16) == 255

def test_getFloat_binary_chromosome():
    size = 128
    chromosome = ch.BinaryChromosome(size)
    chromosome.values = np.zeros(size, dtype=np.dtype(bool))
    assert np.abs(chromosome.getFloat(64, 128, -5, 5) + 5) < 0.001

    chromosome.values = np.ones(size, dtype=np.dtype(bool))
    assert np.abs(chromosome.getFloat(64, 128, -5, 5) - 5) < 0.001

def test_numeric_chromosome_factory():
    size = 8
    minValues = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    maxValues = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    factory = ch.NumericChromosomeFactory(size, minValues, maxValues)
    chromosome = factory.createChromosome()

    assert len(chromosome.values) == size
    assert np.array_equal(chromosome.minValues, minValues)
    assert np.array_equal(chromosome.maxValues, maxValues)

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

def test_renormalize_numeric_chromosome():
    min = -5
    max = np.array([-0.5, 9, 150])
    chromosome = ch.NumericChromosome(3, min, max)
    chromosome.values = np.array([-7, 10, 151])
    chromosome.renormalize()
    assert np.array_equal(chromosome.values, [-5, 9, 150])

def test_copyValues_numeric_chromosome():
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
