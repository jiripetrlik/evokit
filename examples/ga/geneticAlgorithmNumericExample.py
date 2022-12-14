import evokit.chromosome as chr
import evokit.crossover as cr
import evokit.mutation as mu
import evokit.ga as ga

def main():
    fitness = lambda ch : ((ch.values[0] - 1) ** 2) + (ch.values[1] ** 2) + ((ch.values[2] + 1) ** 2)
    chromosomeFactory = chr.NumericChromosomeFactory(3, -5000, 5000)
    crossover = cr.SimulatedBinaryCrossover()
    mutation = mu.NormalDistributionMutation(1)
    result = ga.geneticAlgorithm(fitness, chromosomeFactory, 100, crossover, mutation, 1000)

    print("Best fitness:", result["fitness"])
    print("Best solutions:", result["solution"].values)
    result["observer"].plot()

if __name__ == "__main__":
    main()
