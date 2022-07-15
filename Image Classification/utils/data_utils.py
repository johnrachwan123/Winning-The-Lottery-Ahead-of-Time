import numpy as np
from torch import nn


"""
Just some util functions
"""

def smooth(array, window=5000):
    new_array = np.zeros_like(array)
    for i in range(len(array)):
        begin = 0
        end = len(array) - 1
        if i + window < len(array) - 1:
            end = i + window
        if i - window > 0:
            begin = i - window
        new_array[i] = np.mean(array[begin:end])
    return np.array(new_array)


def fin_diff(array, type=np.ndarray):
    if not isinstance(array, type):
        array = type(array)
    return array[1:] - array[:-1]


def lookahead_type(iterable):
    it = iter(
        list([element for element in iterable if
              isinstance(element, nn.Conv2d) or isinstance(element, nn.Linear) or isinstance(element,
                                                                                             nn.AdaptiveAvgPool2d)]))
    last = next(it)
    for val in it:
        yield last, (isinstance(last, nn.Conv2d), isinstance(val, nn.Conv2d))
        last = val
    yield last, (isinstance(last, nn.Conv2d), None)


def lookahead_finished(iterable):
    it = iter(iterable)
    last = next(it)
    first = True
    for val in it:
        yield last, (first, False)
        last = val
        if first:
            first = False
    yield last, (False, True)
