import evokit.chromosome as chr
import evokit.crossover as cr
import evokit.mutation as mu
import evokit.moea as m

def main():
    fitnessFunctions = [
        lambda ch: (ch.values[0] - 490) ** 2,
        lambda ch: (ch.values[0] - 510) ** 2
    ]
    chromosomeFactory = chr.NumericChromosomeFactory(1, -5000, 5000)
    crossover = cr.SimulatedBinaryCrossover()
    mutation = mu.NormalDistributionMutation(1)
    result = m.nsga2(fitnessFunctions, chromosomeFactory, 50, crossover, mutation, 200)

    print("Best fitness:", result["fitnessValues"])
    print("Best solutions:", result["solutions"])
    m.plotNondominatedSolutions(result["fitnessValues"])
    result["observer"].plot()

if __name__ == "__main__":
    main()
