import numpy as np

def normalDistributionMutation(chromosome, sd = 0.1):
    difference = np.random.normal(scale = sd)
    chromosome.values += difference
    chromosome.renormalize()
