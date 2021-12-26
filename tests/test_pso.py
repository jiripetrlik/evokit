import numpy as np
import evokit.pso as pso

def test_pso():
    np.random.seed(0)
    fitness = lambda v : ((v[0] - 1) ** 2) + (v[1] ** 2) + ((v[2] + 1) ** 2)
    iterations = 200

    for _ in range(5):
        result = pso.pso(fitness, 3, -10, 10, 100, iterations)
        assert result["fitness"] < 0.001
        error = np.abs(result["solution"] - np.array([1, 0, -1]))
        assert np.all(error < 0.01)
        assert len(result["observer"].minFitness) == iterations
        assert len(result["observer"].meanFitness) == iterations
        assert len(result["observer"].maxFitness) == iterations
