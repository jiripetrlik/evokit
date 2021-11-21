import evokit.es as e

def main():
    fitness = lambda v : ((v[0] - 1) ** 2) + (v[1] ** 2) + ((v[2] + 1) ** 2)
    result = e.miPlusOneES(fitness, 3, 8, 100)

    print("Best fitness:", result["fitness"])
    print("Best solutions:", result["solution"])
    result["observer"].plot()

if __name__ == "__main__":
    main()
