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
    #print('##', y, len(y), np.shape(y))
    #print('@@', alpha, np.shape(alpha))
    #print('%%', beta, np.shape(beta))
    #print('***',alpha,beta)
    d_alpha = beta
    d_beta = -b * beta - a * np.sin(alpha)
    return([d_alpha, d_beta])

# Solve(integrate) the ode
y = integrate.odeint(pODE,y0,t,args=(a,b))

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
fig1 = plt.figure(figsize=(12,6))
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

# equilibrium points X_f0 and X_f1
values  = np.linspace(0, 1., 8)

# let's get some fancy colors for each trajectory
vcolors = plt.cm.autumn_r(np.linspace(0.3, 1., len(values)))
fig2 = plt.figure(figsize=(12,6))

# plot trajectories
for v, col in zip(values, vcolors):
    # starting point
    X0 = np.array(v * np.array([np.pi-0.01, 10.]))
    # integrate the ODE for the times and starting points
    X = integrate.odeint(pODE, X0, t ,args=(a,b))
    # plot the trajectory with varying linewidth and color
    plt.plot(X0[0], X0[1], 'xk')
    plt.plot( X[:,0], X[:,1], lw=3.5*v, color=col, label='X0=(%.1f, %.1f)' % ( X0[0], X0[1]) )
    plt.legend(loc=2)

# Make a quiver-plot
nb_points = 20
alpha_domain = np.linspace(-35, 35, nb_points)
beta_domain = np.linspace(-12, 12, nb_points)
#print(alpha_domain, beta_domain)

X, Y = np.meshgrid(alpha_domain, beta_domain)

#print('***',X,Y)
U, V = pODE([X, Y],t,a,b)

# to synchronize the scale of grids between 'trajectories' and 'arrows of the direction field'
# otherwise, the direction of arrows look ** NOT TO BE tangential to the trajectory.**
#V = V * np.max(alpha_domain) / np.max(beta_domain)

M = np.sqrt(U**2+V**2)

# Avoid zero division errors - for example, in case we hit the stability point
M[M == 0] = 1.

# Normalize each arrow by the growth rate's norm
U_Norm = U/M
V_Norm = V/M
#print(type(M),type(U),type(V))

plt.title('Direction field and Some trajectories')
plt.quiver(X, Y, U_Norm, V_Norm, M, pivot='mid', cmap=plt.cm.jet)

plt.show()
