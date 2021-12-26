import argparse
import sys
import evokit.benchmarks as b
import evokit.pso as pso

BENCHMARKS = ["ackley", "ackley-test", "rosenbrock", "sphere"]

def main():
    parser = argparse.ArgumentParser(description="Run PSO for benchmark")
    parser.add_argument("--benchmark", dest="benchmark", required=True, type=str,
        help=f"Benchmark function ({BENCHMARKS})")
    parser.add_argument("--dimension", dest="dimension", required=True, type=int,
        help=f"Number of benchmark dimensions")
    parser.add_argument("--iterations", dest="iterations", required=True, type=int,
        help="Number of PSO iterations")
    parser.add_argument("--plot", dest="plot", action="store_true",
        help="Plot progress of PSO")
    parser.add_argument("--particles", dest="particles", required=True, type=int,
        help="Number of particles")

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

    result = pso.pso(benchmark, args.dimension, minValues, maxValues, args.particles,
                        args.iterations)
    
    print("Fitness: ", result["fitness"])
    print("Solution: ", result["solution"])
    if args.plot == True:
        result["observer"].plot()

if __name__ == "__main__":
    main()
