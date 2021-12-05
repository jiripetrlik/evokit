import numpy as np

class BitFlipMutation:
    def __init__(self, probability = 0.05):
        self.probability = probability

    def mutation(self, chromosome):
        size = len(chromosome.values)

        r = np.random.uniform(size = size)
        for i in range(size):
            if r[i] < self.probability:
                if chromosome.values[i] == True:
                    chromosome.values[i] = False
                else:
                    chromosome.values[i] = True

class NormalDistributionMutation:
    def __init__(self, sd = 0.1):
        self.sd = sd
        
    def mutation(self, chromosome):
        difference = np.random.normal(scale = self.sd, size=len(chromosome.values))
        chromosome.values += difference
        chromosome.renormalize()

class ReciprocalExchangeMutation:
    def mutation(self, chromosome):
        size = len(chromosome.values)
        index1 = np.random.randint(size)
        index2 = np.random.randint(size)

        tmp = chromosome.values[index1]
        chromosome.values[index1] = chromosome.values[index2]
        chromosome.values[index2] = tmp
