import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib import animation

# Define Constants and Initial conditions
y0 = np.array([np.pi - 0.01, 0]) # init of alpha and beta
t = np.linspace(0, 50, 1001) # we upgraded steps to make it smoothly
# t = np.linspace(0, 50, 101)

a = 5                    # coefficient of friction
b = 0.25                 # coefficient of gravity

# Define ODE
# alpha'(t) = beta(t)
# beta'(t) = -b*beta(t) - a*sin(alpha(t))

def pODE(y,t,a,b):
    alpha, beta = y
    d_alpha = beta
    d_beta = -b * beta - a * np.sin(alpha)
    return([d_alpha, d_beta])

# Solve(integrate) the ode
y = integrate.odeint(pODE,y0,t,args=(a,b))

# Extract alpha and beta from y
y_alpha = y[:,0]
y_beta = y[:,1]

# Find the x-intercept of alpha to find the moment(time) that angular velocity is maximum
x_i = 0
for i in range(len(y_alpha)):
    if y_alpha[i] >= 0 and y_alpha[i+1] < 0:
        x_i = t[i]
        break

# Plots of alpha(t) and beta(t)
fig1 = plt.figure(figsize=(12,8))
plt.plot(t,y_alpha,'b--',t,y_beta,'r-')
plt.axvline(x=x_i)
plt.plot(t,[0]*len(y_alpha),'k')
plt.legend(['alpha : the angle of deflection','beta : the angular speed','Time of Maximum speed'])
plt.xlabel('time [s]')
plt.ylabel('state')
plt.title('Plots of alpha(t) and beta(t)')
fig1.savefig('pendulum_1.png')

''' Comment 1
As shown in the plot, when the pendulum pass the origin(alpha == 0) for the first time,
the speed(magnitude of velocity) becomes Maximum value.
'''

# Get some fancy colors for each trajectory
values  = np.linspace(0, 1., 8)
vcolors = plt.cm.autumn_r(np.linspace(0.3, 1., len(values)))

fig2 = plt.figure(figsize=(12,8))

# Plot the trajectories for 8 different starting points
for v, col in zip(values, vcolors):
    # 8 starting points
    X0 = np.array(v * np.array([np.pi-0.01, 10.]))
    # integrate the ODE for the times and starting points
    X = integrate.odeint(pODE, X0, t ,args=(a,b))
    # plot the trajectory with varying linewidth and color
    plt.plot(X0[0], X0[1], 'xk')
    plt.plot( X[:,0], X[:,1], lw=2*v+1, color=col, label='X%.f=(%.1f, %.1f)' % (np.where(values==v)[0], X0[0], X0[1]) )
    plt.legend(loc=2)

    # get animation data for X1 & X7
    if np.where(values==v)[0] == 1:        slowest_alpha = X[:,0]
    if np.where(values==v)[0] == 7:        fastest_alpha = X[:,0]

# Make a quiver-plot
nb_points = 20
alpha_domain = np.linspace(-35, 35, nb_points)
beta_domain = np.linspace(-12, 12, nb_points)

X, Y = np.meshgrid(alpha_domain, beta_domain)
U, V = pODE([X, Y],t,a,b)
M = np.sqrt(U**2+V**2)

# Avoid zero division errors - for example, in case we hit the stability point
M[M == 0] = 1.

# Normalize each arrow by the growth rate's norm
U_Norm = U/M
V_Norm = V/M

plt.title('Direction field and Some trajectories')
plt.quiver(X, Y, U_Norm, V_Norm, M, pivot='mid', cmap=plt.cm.jet)
fig2.savefig('pendulum_2.png')

''' Comment 2
 => Qualitatively describe the motion of the pendulum for the fastest versus the second-slowest trajectory
    : the fastest one(X7 in picture) initially have enough speed(kinetic energy) to go round and round for Four-times.
    but at fifth turn, because of the friction, the speed comes down, which means that kinetic energy get lost.
    eventually the pendulum stop to turn around, and move back and forth in the direction of angular velocity.

    on the other hand, the second-slowest one(X1 in fig) don't have enough speed go round even in the first try.
    so it goes back and forth and become to stay after some time.
'''

#fig3, axis = plt.subplots()
fig3, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_xlim([-10, 10])
ax2.set_xlim([-10, 10])
ax1.set_title('the second-slowest pendulum')
ax2.set_title('the fastest pendulum')
plt.ylim(-10,10)
line1, = ax1.plot([], [], 'ro')
line2, = ax2.plot([], [], 'bo')

def init_1():
    line1.set_data([], [])
    return line1,

def init_2():
    line2.set_data([], [])
    return line2,

def ani_slow(i):
    x = [5*np.sin(slowest_alpha[i])]
    y = [-5*np.cos(slowest_alpha[i])]
    line1.set_data(x, y)
    return line1,

def ani_fast(i):
    x = [5*np.sin(fastest_alpha[i])]
    y = [-5*np.cos(fastest_alpha[i])]
    line2.set_data(x, y)
    return line2,

movie1 = animation.FuncAnimation(fig3, ani_slow, init_func=init_1, frames=len(t), interval=40)#, blit=True)
movie2 = animation.FuncAnimation(fig3, ani_fast, init_func=init_2, frames=len(t), interval=40)#, blit=True)

plt.show()
