import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Define Constants and Initial conditions
y0 = np.array([np.pi - 0.01, 0]) # init of alpha and beta
# t = np.linspace(0, 50, 101)
t = np.linspace(0, 50, 1001)

a = 5                    # coefficient of friction
b = 0.25                 # coefficient of gravity

# Define ODE
def pODE(y,t,a,b):
    # alpha'(t) = beta(t)
    # beta'(t) = -b*beta(t) - a*sin(alpha(t))
    alpha, beta = y
    d_alpha = beta
    d_beta = -b * beta - a * np.sin(alpha)
    return([d_alpha, d_beta])

# Solve(integrate) the ode
y = integrate.odeint(pODE,y0,t,(a,b))

# extract alpha and beta from y
y_alpha = y[:,0]
y_beta = y[:,1]

# find x-intercept of alpha to maximize angular velocity
x_i = 0
for i in range(len(y_alpha)):
    if y_alpha[i] >= 0 and y_alpha[i+1] < 0:
        x_i = t[i]
        break

# Plots
fig = plt.figure(figsize=(12,6))
plt.plot(t,y_alpha,'b--',t,y_beta,'r-')
plt.axvline(x=x_i)
plt.plot(t,[0]*len(y_alpha),'k')
plt.legend(['alpha : the angle of deflection','beta : the angular speed','Maximum Angular Velocity'])
plt.xlabel('time [s]')
plt.ylabel('state')

''' Comment 1
As shown in the plot, when the pendulum pass the origin for the first time, the speed(magnitude of velocity)
become Maximum value.
'''

plt.show()