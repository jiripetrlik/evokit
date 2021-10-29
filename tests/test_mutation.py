import numpy as np
import evokit.chromosome as chr
import evokit.mutation as mu

def test_normalDistributionMutation():
    np.random.seed(0)
    size = 5
    chromosome = chr.NumericChromosome(size, minValues=1, maxValues=5)
    for _ in range(100):
        chromosome.initialize()
        originalValues = np.copy(chromosome.values)
        mu.normalDistributionMutation(chromosome)
        for i in range(size):
            assert chromosome.values[i] != originalValues[i]
            assert chromosome.values[i] >= chromosome.minValues
            assert chromosome.values[i] <= chromosome.maxValues
