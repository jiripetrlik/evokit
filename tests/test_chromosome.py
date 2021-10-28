import evokit.chromosome

def test_initialize_numeric_chromosome():
    chromosome = evokit.chromosome.NumericChromosome(5, 1, 7)
    chromosome.initialize()
    for i in range(len(chromosome.values)):
        assert chromosome.values[i] >= 1
        assert chromosome.values[i] <= 7
