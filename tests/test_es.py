import evokit.es as e
import numpy as np

def test_onePlusOneES():
    np.random.seed(0)
    fitness = lambda v : ((v[0] - 1) ** 2) + (v[1] ** 2) + ((v[2] + 1) ** 2)
    iterations = 100
    
    for _ in range(5):
        result = e.onePlusOneES(fitness, 3, 0.1, iterations)
        assert result["fitness"] < 0.01
        error = result["solution"] - np.array([1, 0, -1])
        assert np.all(error < 0.1)
        assert len(result["observer"].minFitness) == iterations
        assert len(result["observer"].improved) == iterations

def test_miPlusOneES():
    np.random.seed(0)
    fitness = lambda v : ((v[0] - 1) ** 2) + (v[1] ** 2) + ((v[2] + 1) ** 2)
    iterations = 1000

    for _ in range(5):
        result = e.miPlusOneES(fitness, 3, 8, iterations)
        assert result["fitness"] < 0.1
        error = result["solution"] - np.array([1, 0, -1])
        assert np.all(error < 0.1)
        assert len(result["observer"].minFitness) == iterations
        assert len(result["observer"].meanFitness) == iterations
        assert len(result["observer"].maxFitness) == iterations
        assert len(result["observer"].improved) == iterations

def test_miPlusLambdaES():
    np.random.seed(0)
    fitness = lambda v : ((v[0] - 1) ** 2) + (v[1] ** 2) + ((v[2] + 1) ** 2)
    iterations = 100

    for _ in range(5):
        result = e.miPlusLambdaES(fitness, 3, 8, 16, iterations)
        assert result["fitness"] < 0.1
        error = result["solution"] - np.array([1, 0, -1])
        assert np.all(error < 0.1)
        assert len(result["observer"].minFitness) == iterations
        assert len(result["observer"].meanFitness) == iterations
        assert len(result["observer"].maxFitness) == iterations
        assert len(result["observer"].improved) == iterations
