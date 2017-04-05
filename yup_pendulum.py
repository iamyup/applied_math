# Assignment 2 - secon task [40 points]

import math
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

# original alpha
def alpha(t):
    return a*np.sin(t)

# The angular speed is simply the time-derivative of the angle
def beta(t):
    return a*np.cos(t)


def alpha_prime(t):
    return beta(t)


def alpha_double_prime(y, t, a, b):
    return -a * np.sin(y) - b * beta(t)

a = 1.0
b = 1.0
t = np.linspace(0.0, 10*np.pi, 1000)
y = alpha(t)
# y_double_prime = integrate.odeint(alpha_double_prime, y, t, args=(a,))
y_double_prime = alpha_double_prime(y, t, a, b)

# plt.plot(t, y_double_prime)
plt.plot(t, y, t, y_double_prime)
plt.grid()
plt.show()


