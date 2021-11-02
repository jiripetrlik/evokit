import evokit.chromosome as chr
import evokit.crossover as cr
import evokit.mutation as mu
import evokit.ga as ga

def fitness(chromosome):
    x = chromosome.getFloat(0, 32, -5000, 5000)
    y = chromosome.getFloat(32, 64, -5000, 5000)
    z = chromosome.getFloat(64, 96, -5000, 5000)

    value = ((x - 1) ** 2) + (y ** 2) + ((z + 1) ** 2)
    return value

def main():
    chromosomeFactory = chr.BinaryChromosomeFactory(96)
    crossover = cr.UniformCrossover()
    mutation = mu.BitFlipMutation(0.01)
    result = ga.geneticAlgorithm(fitness, chromosomeFactory, 100,
                crossover, mutation, 1000)

    print("Best fitness:", result["fitness"])
    print("Best solution:",
            result["solution"].getFloat(0, 32, -5000, 5000),
            result["solution"].getFloat(32, 64, -5000, 5000),
            result["solution"].getFloat(64, 96, -5000, 5000))
    result["observer"].plot()

if __name__ == "__main__":
    main()
