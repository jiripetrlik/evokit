import argparse
import evokit.chromosome as ch
import evokit.crossover as c
import evokit.moea as moea
import evokit.moeabenchmarks as b
import evokit.mutation as m
import sys

ALGORITHMS = ["vega", "nsga2"]
BENCHMARKS = ["sch", "fon", "zdt1", "zdt2", " zdt3"]

def main():
    parser = argparse.ArgumentParser(description="Run multiobjective genetic algorithm for benchmark")
    parser.add_argument("--algorithm", dest="algorithm", required=True, type=str,
        help=f"Multiobjective genetic algorithm ({ALGORITHMS})")
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

    if args.benchmark == "sch":
        minValues = b.SCH_MIN
        maxValues = b.SCH_MAX
        benchmark = b.schProblem
    elif args.benchmark == "fon":
        minValues = b.FON_MIN
        maxValues = b.FON_MAX
        benchmark = b.fonProblem
    elif args.benchmark == "zdt1":
        minValues = b.ZDT1_MIN
        maxValues = b.ZDT1_MAX
        benchmark = b.zdt1Problem
    elif args.benchmark == "zdt2":
        minValues = b.ZDT2_MIN
        maxValues = b.ZDT2_MAX
        benchmark = b.zdt2Problem
    elif args.benchmark == "zdt3":
        minValues = b.ZDT3_MIN
        maxValues = b.ZDT3_MAX
        benchmark = b.zdt3Problem
    else:
        print("Unknown benchmark function: ", args.benchmark)
        sys.exit(1)

    chromosomeFactory = ch.NumericChromosomeFactory(args.dimension, minValues, maxValues)
    crossover = c.SimulatedBinaryCrossover()
    mutation = m.NormalDistributionMutation()

    if args.algorithm == "vega":
        result = moea.vega(lambda chromosome: benchmark(chromosome.values),
            chromosomeFactory, args.population, crossover, mutation, args.iterations)
    elif args.algorithm == "nsga2":
        result = moea.nsga2(lambda chromosome: benchmark(chromosome.values),
            chromosomeFactory, args.population, crossover, mutation, args.iterations)
    else:
        print("Unknown multiobjective genetic algorithm", args.algorithm)

    print("Best fitness:", result["fitnessValues"])
    print("Best solutions:", result["solutions"])
    if args.plot == True:
        result["observer"].plot()

if __name__ == "__main__":
    main()
