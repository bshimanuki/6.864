import scipy as sp


def sigmoid(steepness, midpoint):
    return lambda x: sp.special.expit(steepness * (x - midpoint))