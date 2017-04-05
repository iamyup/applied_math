from scipy import integrate
import numpy as np
import matplotlib.pylab as plt

# logistic equation
def dp_dt(P, t=0, A=0.08, K=1000):
    return (A * P * (1 - P/K))


#default parameter
A = 0.08
K = 1000

# total time we want to get our data for
t = np.linspace(0, 100, 10000)

# initial function value for two cases
f0_increase = 10
f0_decrease = 1500

# let's use the integration method of scipy
# note that "A", "K" is passed as a tuple to scipy
# function value for two cases
ft_increase = integrate.odeint(dp_dt, f0_increase, t, args=(A, K,))
ft_decrease = integrate.odeint(dp_dt, f0_decrease, t, args=(A, K,))

# plot all possible solutions
fig = plt.figure()
plt.plot(t, ft_increase, t, ft_decrease)

# get axis limits
ymax = plt.ylim(ymin=0)[1]
xmax = plt.xlim(xmin=0)[1]
nb_points = 30

x = np.linspace(0, xmax, nb_points)
y = np.linspace(0, ymax, nb_points)

# create a grid with the axis limits
X, Y = np.meshgrid(x, y)

# compute growth rate on the grid
U = 1
V = dp_dt(Y, t, A, K)

# to synchronize the scale of grid between solution plot ft_increase, ft_derease and direction fields
# otherwise, the direction of arrows do not look to be tangential to the solution plot.
V = V * xmax / ymax

# norm of logistics
M = np.sqrt(U**2 + V**2, dtype=np.float64)
# avoid zero division errors
M[M == 0] = 1.

# normalize each arrow
U /= M
V /= M

# plot the direction field for the range of parameters
# define a grid and compute direction at each point
Q = plt.quiver(X, Y, U, V, M, pivot='mid', cmap=plt.cm.jet)

def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    # Update Plot
    fig.canvas.draw()




fig.canvas.mpl_connect('button_press_event', onclick)

# plot carrying capacity & grid
plt.plot([0,100],[1000,1000], 'k--')
plt.grid()
# plt.ylim(0,1600)
plt.show()

