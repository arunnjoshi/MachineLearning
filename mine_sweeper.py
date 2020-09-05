import numpy as np
from pandas.core.generic import Level

"""
    this is mine sweeper game logic .
    mine sweeper is game came in windows 7.
    we pass the loc of cell click by user in start_game function and it return the 'out' if mine is under cell,
    else return nearest mine cell location
"""


def check(arr, x, y):
    try:
        if arr[x, y] == 1:
            return True
        return False
    except:
        return False


def check_neighbour(arr, x, y, level):
    left = check(arr, x - level, y)  # check left
    right = check(arr, x + level, y)  # check right
    top = check(arr, x, y - level)  # top
    bottom = check(arr, x, y + level)  # bootom
    if left or right or top or bottom:
        return True
    else:
        return False


def check_level(arr, x, y, level=1):
    if level > 4:
        return False
    elif check_neighbour(arr, x, y, level):
        return level
    else:
        return check_level(arr, x, y, level + 1)


def start_game(arr, x, y):
    if arr[x, y] == 1:
        return "out"
    return check_level(arr, x, y)


arr = np.random.choice([0], size=(8, 8))
arr[3, 0] = 1
print(arr)
print(start_game(arr, 3, 3))
