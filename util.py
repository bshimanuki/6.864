from scipy.special import expit


def sigmoid(ramp_length, midpoint):
    return lambda x: expit((x - midpoint) / ramp_length)
