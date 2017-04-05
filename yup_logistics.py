from scipy import integrate
import numpy as np
import matplotlib.pylab as plt


# logistic equation
def dp_dt(P, t, A, K):
    return (A * P * (1 - P/K))


#default parameter
A = 0.08
K = 1000

# total time we want to get our data for
t = np.linspace(0, 100, 100)

P = np.linspace(0, 1500, 100)
# let's use the integration method of scipy
# note that "A", "K" is passed as a tuple to scipy
ft = integrate.odeint(dp_dt, P, t, args=(A, K,))

# plot the direction field for the range of parameters
plt.plot(t, ft)
#plt.plot(t, np.exp(t),'r--')
plt.plot([0,100],[1000,1000], 'k--')
plt.grid()
plt.ylim(0,1600)
plt.show()
