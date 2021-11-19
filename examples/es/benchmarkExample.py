import argparse
import evokit.benchmarks as b
import evokit.es as e
import sys

BENCHMARKS = ["ackley", "ackley-test", "rosenbrock", "sphere"]
STRATEGIES = ["oneplusone"]

def main():
    parser = argparse.ArgumentParser(description="Run evolution strategy for benchmark")
    parser.add_argument("--benchmark", dest="benchmark", required=True, type=str,
        help=f"Benchmark function ({BENCHMARKS})")
    parser.add_argument("--dimension", dest="dimension", required=True, type=int,
        help=f"Number of benchmark dimensions")
    parser.add_argument("--iterations", dest="iterations", required=True, type=int,
        help="Number of ES iterations")
    parser.add_argument("--strategy", dest="strategy", required=True, type=str,
        help=f"Type of ES (one of: {STRATEGIES})")

    args = parser.parse_args()

    if args.benchmark == "ackley":
        benchmark = b.ackleyFunction
    elif args.benchmark == "ackley-test":
        benchmark = b.ackleyTestFunction
    elif args.benchmark == "rosenbrock":
        benchmark = b.rosenbrockFunction
    elif args.benchmark == "sphere":
        benchmark = b.sphereFunction
    else:
        print("Unknown benchmark function: ", args.benchmark)
        sys.exit(1)

    if args.strategy == "oneplusone":
        result = e.onePlusOneES(benchmark, args.dimension, 0.1, args.iterations)
    else:
        print("Unknown evolution strategy: ", args.strategy)
        sys.exit(1)

    print(result)

if __name__ == "__main__":
    main()
