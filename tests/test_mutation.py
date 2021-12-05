import numpy as np
import evokit.chromosome as chr
import evokit.mutation as mu

def test_bitFlipMutation():
    np.random.seed(0)
    size = 1024
    mutation = mu.BitFlipMutation()

    chromosome = chr.BinaryChromosome(size)
    chromosome.initialize()
    for _ in range(100):
        oldValues = np.copy(chromosome.values)
        mutation.mutation(chromosome)
        assert len(chromosome.values) == size
        assert np.array_equal(chromosome.values, oldValues) == False

def test_normalDistributionMutation():
    np.random.seed(0)
    size = 5
    chromosome = chr.NumericChromosome(size, minValues=1, maxValues=5)
    mutation = mu.NormalDistributionMutation()
    for _ in range(100):
        chromosome.initialize()
        originalValues = np.copy(chromosome.values)
        mutation.mutation(chromosome)
        for i in range(size):
            assert chromosome.values[i] != originalValues[i]
            assert chromosome.values[i] >= chromosome.minValues
            assert chromosome.values[i] <= chromosome.maxValues

def test_reciprocalExchangeMutation():
    np.random.seed(0)
    size = 256

    changed = 0
    mutation = mu.ReciprocalExchangeMutation()
    for _ in range(100):
        chromosome = chr.PermutationChromosome(size)
        chromosome.initialize()
        originalValues = np.copy(chromosome.values)
        
        mutation.mutation(chromosome)
        assert len(np.unique(chromosome.values)) == size
        if not np.array_equal(chromosome.values, originalValues):
            changed += 1

    assert changed > 0
