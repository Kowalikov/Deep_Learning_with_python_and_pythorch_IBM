from numpy import exp as exp
import math
import scipy


def sigmoid(x):
    return (1/(1+exp(-x)))


z = sigmoid(-4)
print(z)

