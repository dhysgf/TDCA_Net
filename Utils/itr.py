import math
def itr(n, p, t):
    if not all(isinstance(i, (int, float)) for i in [n, p, t]):
        raise TypeError("Inputs must be numeric.")
    if p < 0 or p > 1:
        raise ValueError("Accuracy needs to be between 0 and 1.")
    elif p < 1 / n:
        print("Warning: The ITR might be incorrect because the accuracy < chance level.")
        itr = 0
    elif p == 1:
        itr = math.log2(n) * 60 / t
    else:
        itr = (math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))) * 60 / t
    return itr


