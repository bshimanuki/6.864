from scipy.special import expit


def sigmoid(steepness, midpoint):
    return lambda x: expit(steepness * (x - midpoint))
