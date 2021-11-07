import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

SPHERE_FUNCTION_MIN = -5.12
SPHERE_FUNCTION_MAX = 5.12

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
