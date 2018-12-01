import numpy as np

#def euclidean(a, b):
#    return np.sum((a-b)**2)

#def euclidean2(a, b):
#    return np.sqrt(np.sum((a-b)**2))


def euclidean(a, b):
    distance = 0
    for ai, bi in zip(a,b):
        if type(ai) in [float, np.float64]:
            distance = distance + (ai-bi)**2
        else:
            if ai != bi:
                distance = distance + 1
    return distance


def manhattan(a, b):
    distance = 0
    for ai, bi in zip(a,b):
        if type(ai) in [float, np.float64]:
            distance = distance + abs(ai-bi)
        else:
            if ai != bi:
                distance = distance + 1
    return distance


def cosine(a, b):
    distance_nom = 0
    distance_den1 = 0
    distance_den1 = 0
    for ai, bi in zip(a,b):
        if type(ai) in [float, np.float64]:
            distance_nom = distance_nom + (ai*bi)
            distance_den1 = distance_den1 + ai**2
            distance_den2 = distance_den2 + bi**2
        else:
            if ai != bi:
                distance = distance + 1
    distance = distance_nom / (distance_den1*distance_den2)
    return distance


def canberra(a, b):
    distance = 0
    for ai, bi in zip(a,b):
        if type(ai) in [float, np.float64]:
            distance = distance + abs(ai-bi)/(abs(ai)+abs(bi))
        else:
            if ai != bi:
                distance = distance + 1
    return distance


# Clark ¿?

# HVDM ¿?
