from scipy.special import expit


def sigmoid(ramp_length, midpoint):
    return lambda x: expit((x - midpoint) / ramp_length)

def merge_dicts(args):
    # Dicts must have the same keys
    merged = {}
    for key in args[0]:
        for i, arg in enumerate(args):
            merged[key + str(i)] = arg[key]
    return merged
