import numpy as np

SCH_MIN = -1000
SCH_MAX = 1000
FON_MIN = -4
FON_MAX = 4
ZDT1_MIN = 0
ZDT1_MAX = 1
ZDT2_MIN = 0
ZDT2_MAX = 1
ZDT3_MIN = 0
ZDT3_MAX = 1

def schProblem(inputs):
    v = np.array(inputs)
    return v[0] ** 2, (v[0] - 2) ** 2

def fonProblem(inputs):
    v = np.array(inputs)
    o1 = 1 - np.exp(np.sum((v - 1 / np.sqrt(3)) ** 2))
    o2 = 1 - np.exp(np.sum((v + 1 / np.sqrt(3)) ** 2))

    return o1, o2

def zdt1Problem(inputs):
    v = np.array(inputs)
    o1 = v[0]
    g = 1 + 9 * np.sum(v[1:]) / (len(v) - 1)
    o2= g * (1 - np.sqrt(v[0] / g))

    return o1, o2

def zdt2Problem(inputs):
    v = np.array(inputs)
    o1 = v[0]
    g = 1 + 9 * np.sum(v[1:]) / (len(v) - 1)
    o2 = g * (1 - (v[0] / g) ** 2)

    return o1, o2

def zdt3Problem(inputs):
    v = np.array(inputs)
    o1 = v[0]
    g = 1 + 9 * np.sum(v[1:]) / (len(v) - 1)
    o2 = g * (1 - np.sqrt(v[0] / g) - (v[0] / g) * np.sin(10 * np.pi * v[0]))

    return o1, o2
