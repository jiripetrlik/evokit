import evokit.moea as m

def test_weightedFitness():
    weights = [2, 7, 11]
    fitnessFunctions = [
        lambda x: x[0] ** 2 + 5,
        lambda x: x[0] - 7,
        lambda x: 2 * x[0] - 1
    ]
    weightedFitnessFunction = m.WeightedSumFitness(weights, fitnessFunctions)

    assert weightedFitnessFunction([2]) == 16
    assert weightedFitnessFunction([5]) == 145
    assert weightedFitnessFunction([-2]) == -100
