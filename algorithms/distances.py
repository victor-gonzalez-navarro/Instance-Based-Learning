import numpy as np
import math


def euclidean(a, b):
    distance = 0
    for ai, bi in zip(a,b):
        if type(ai) in [float, np.float64]:
            distance = distance + (ai-bi)**2
        else:
            if ai != bi or ai == b'?' or bi == b'?':
                distance = distance + 1
    return distance


def manhattan(a, b):
    distance = 0
    for ai, bi in zip(a,b):
        if type(ai) in [float, np.float64]:
            distance = distance + abs(ai-bi)
        else:
            if ai != bi or ai == b'?' or bi == b'?':
                distance = distance + 1
    return distance


def canberra(a, b):
    distance = 0
    for ai, bi in zip(a,b):
        if type(ai) in [float, np.float64]:
            distance = distance + abs(ai-bi)/abs(ai+bi)
        else:
            if ai != bi or ai == b'?' or bi == b'?':
                distance = distance + 1
    return distance


def hvdm(a, b, trn_data, trn_labels):
    distance = 0
    feature_number = 0
    labels, counts = np.unique(trn_labels, return_counts=True)
    num_clases = len(counts)
    for ai, bi in zip(a,b):
        if type(ai) in [float, np.float64]:
            distance = distance + (ai - bi)**2
        else:
            if ai == b'?' or bi == b'?':
                distance = distance + 1
            elif ai != bi:
                distance = distance + normalized_vdm1(trn_data, trn_labels, feature_number, ai, bi, num_clases)
        feature_number = feature_number + 1
    return distance


def normalized_vdm1(trn_data, trn_labels, feature_number, ai, bi, num_clases):
    Nax = np.sum(trn_data[:,feature_number] == ai)
    Nay = np.sum(trn_data[:,feature_number] == bi)
    Naxc = 0.0; Nayc = 0.0; aux_distance = 0.0

    for c in range(num_clases):
        Naxc = Naxc + np.sum((trn_data[:,feature_number] == ai) * (trn_labels == c))
        Nayc = Nayc + np.sum((trn_data[:,feature_number] == bi) * (trn_labels == c))
        aux_distance = aux_distance + abs((Naxc/Nax)-(Nayc/Nay))

    return aux_distance
