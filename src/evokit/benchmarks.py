import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

ACKLEY_FUNCTION_MIN = -30
ACKLEY_FUNCTION_MAX = 30
ACKLEY_TEST_FUNCTION_MIN = -30
ACKLEY_TEST_FUNCTION_MAX = 30
ROSENBROCK_FUNCTION_MIN = -2.048
ROSENBROCK_FUNCTION_MAX = 2.048
SPHERE_FUNCTION_MIN = -5.12
SPHERE_FUNCTION_MAX = 5.12

def ackleyFunction(x):
    denominator = np.arange(start = 1, stop = len(x) + 1)
    return (20 + np.e - 20 * np.exp(-0.2 * np.sum(x ** 2 / denominator))
                - np.exp(np.sum(np.cos(2 * np.pi * x) / denominator)))

def ackleyTestFunction(x):
    v1 = x[:-1]
    v2 = x[1:]
    return np.sum(3 * (np.cos(2 * v1) + np.sin(2 * v2)) 
                + np.exp(-0.2) * np.sqrt(v1 ** 2 + v2 ** 2))

def rosenbrockFunction(x):
    v1 = x[:-1]
    v2 = x[1:]

    return np.sum(100 * (v2 - v1 ** 2) ** 2 + (v1 - 1) ** 2)

def sphereFunction(x):
    x = np.array(x)
    return np.sum(x ** 2)

def plotBenchmark(benchmark, min = -10, max = 10):
    x = np.linspace(start = min, stop = max, num = 150, dtype=np.float64)
    y = np.linspace(start = min, stop = max, num = 150, dtype=np.float64)
    x, y = np.meshgrid(x, y)
    
    z = np.zeros(np.shape(x), dtype=np.float64)
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            z[i, j] = benchmark(np.array([x[i, j], y[i, j]]))

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.viridis)

    plt.show()
