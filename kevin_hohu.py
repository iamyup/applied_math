import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sys

print('.....MODE :',sys.argv)

# determine ext_I from sys.argv
I_all = np.array([[1e-5,1.5,2,2.5,3,4,5,10,15,20,50],[15.0],[15.0],[10,15,20,50],[0],[0]])
I = I_all[int(sys.argv[1])-1]

# Maximal conductances as np.array
# 1=potassium(K),2=sodium(Na), 3=leakage
g = np.array([36, 120, 0.3])

# Battery voltages as np.array
# 1=n-particle, 2=m-particle, 3=h-particle
E = np.array([-12, 115, 10.613])

# time for integration (in ms)
# note that we start at negative times to allow the membrane to relax
# its dynamics, before we start plotting at t=0
time = np.linspace(-30, 50, 8001)

# starting values for V,n,m,h:
s0 = np.array([-10, 0, 0, 0])
fig1 = plt.figure(figsize=(15,8))

for ii in I:
    def iext(t):
        if sys.argv[1] == '1':
            #print('start MODE :',sys.argv[1])
            if t >= 10 and t < 12: iext = ii
            else: iext = 0
        elif sys.argv[1] == '2':
            if t >= 10 and t < 12: iext = ii
            elif t >= 19 and t < 21: iext = ii
            else: iext = 0
        elif sys.argv[1] == '3':
            if t >= 10 and t < 12: iext = ii
            elif t >= t0 and t < t0+2: iext = ii
            else: iext = 0
        elif sys.argv[1] == '4':
            if t >= 10 and t < 40: iext = ii
            else: iext = 0
        elif sys.argv[1] == '5':
            iext = 0
        else:
            print('Please input the args from 1 to 5')
            iext = 0
        return(iext)

    def ode(X, t):
        V, m, n, h = X
        # slide 66
        dV_dt = iext(t) - (g[0]*n**4*(V-E[0]) + g[1]*m**3*h*(V-E[1]) + g[2]*(V-E[2]))
        # slides 64,65
        alphan = (0.1-0.01*V)/(np.exp(1-0.1*V)-1)
        alpham = (2.5-0.1*V)/(np.exp(2.5-0.1*V)-1)
        alphah = 0.07 * np.exp(-V/20)
        betan = 0.125 * np.exp(-V/80)
        betam = 4 * np.exp(-V/18)
        betah = 1 / (np.exp(3-0.1*V)+1)
        # slide 66
        dm_dt = alpham * (1-m) - betam * m
        dn_dt = alphan * (1-n) - betan * n
        dh_dt = alphah * (1-h) - betah * h

        return ([dV_dt,dm_dt,dn_dt,dh_dt])

    # Solve(integrate) the ode
    y = integrate.odeint(ode,s0,time)

    # Extract V, m, n, h from y
    y_V = y[:, 0]
    y_m = y[:, 1]
    y_n = y[:, 2]
    y_h = y[:, 3]
    plt.plot(time,y_V)

#First plot: the solution V(t) plotted from t=0:end INCLUDING the current injection time points clearly marked (shaded region, box, etc.) using plt.ylim([-20,120])

#Second plot: the solutions n,m,h plotted from t=0:end INCLUDING the current injection time points clearly marked (shaded region, box, etc.)

plt.ylim([-20,120])
plt.show()