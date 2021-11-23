import argparse
import evokit.benchmarks as b
import evokit.chromosome as ch
import evokit.crossover as c
import evokit.mutation as m
import evokit.ga as g
import sys

BENCHMARKS = ["ackley", "ackley-test", "rosenbrock", "sphere"]

def main():
    parser = argparse.ArgumentParser(description="Run genetic algorithm for benchmark")
    parser.add_argument("--benchmark", dest="benchmark", required=True, type=str,
        help=f"Benchmark function ({BENCHMARKS})")
    parser.add_argument("--dimension", dest="dimension", required=True, type=int,
        help=f"Number of benchmark dimensions")
    parser.add_argument("--iterations", dest="iterations", required=True, type=int,
        help="Number of GA iterations")
    parser.add_argument("--plot", dest="plot", action="store_true",
        help="Plot progress of evolution")
    parser.add_argument("--population", dest="population", required=True, type=int,
        help="Population size")
    
    args = parser.parse_args()

    if args.benchmark == "ackley":
        minValues = b.ACKLEY_FUNCTION_MIN
        maxValues = b.ACKLEY_FUNCTION_MAX
        benchmark = b.ackleyFunction
    elif args.benchmark == "ackley-test":
        minValues = b.ACKLEY_TEST_FUNCTION_MIN
        maxValues = b.ACKLEY_TEST_FUNCTION_MAX
        benchmark = b.ackleyTestFunction
    elif args.benchmark == "rosenbrock":
        minValues = b.ROSENBROCK_FUNCTION_MIN
        maxValues = b.ROSENBROCK_FUNCTION_MAX
        benchmark = b.rosenbrockFunction
    elif args.benchmark == "sphere":
        minValues = b.SPHERE_FUNCTION_MIN
        maxValues = b.SPHERE_FUNCTION_MAX
        benchmark = b.sphereFunction
    else:
        print("Unknown benchmark function: ", args.benchmark)
        sys.exit(1)

    chromosomeFactory = ch.NumericChromosomeFactory(args.dimension, minValues, maxValues)
    crossover = c.SimulatedBinaryCrossover()
    mutation = m.NormalDistributionMutation()
    result = g.geneticAlgorithm(lambda chromosome: benchmark(chromosome.values),
        chromosomeFactory, args.population, crossover, mutation, args.iterations)

    print("Fitness: ", result["fitness"])
    print("Solution: ", result["solution"].values)
    if args.plot == True:
        result["observer"].plot()

if __name__ == "__main__":
    main()
