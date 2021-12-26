import evokit.pso as pso

def main():
    fitness = lambda v : ((v[0] - 1) ** 2) + (v[1] ** 2) + ((v[2] + 1) ** 2)
    result = pso.pso(fitness, 3, -10, 10, 100, 100)

    print("Best fitness:", result["fitness"])
    print("Best solutions:", result["solution"])
    result["observer"].plot()

if __name__ == "__main__":
    main()
