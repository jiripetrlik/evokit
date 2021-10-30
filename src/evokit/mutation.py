import numpy as np

class NormalDistributionMutation:
    def __init__(self, sd = 0.1):
        self.sd = sd
        
    def mutation(self, chromosome):
        difference = np.random.normal(scale = self.sd)
        chromosome.values += difference
        chromosome.renormalize()
