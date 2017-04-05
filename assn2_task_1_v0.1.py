import numpy as np
import matplotlib.pyplot as plt

#define the DE as dP/dt = A * P * (1 - P/K)
def dy_dt(A, y, K):
    dy_dt = A * y * (1 - y / K)
    return (dy_dt)

# step size for Euler integration
dt = 0.01

# total number of integration steps
totNumSteps = int(100 / dt)
#print('Total Step',totNumSteps)

# define arrays
y1 = np.zeros(totNumSteps)
y2 = np.zeros(totNumSteps)
t = np.zeros(totNumSteps)

# since Euler's method is iterative, we need starting
# values for both ft and t
# think about what these are and put them in:

y1[0] = 10
y2[0] = 1500
t[0] = 0

# i.e. A=0.08, K=1000 as default parameters.
# good, now here we actually do the numerical integration
# so, we need a for-loop that goes through all the steps:

A = 0.08
K = 1000
for i in np.arange(1, totNumSteps):
    y1[i] = y1[i - 1] + dt * dy_dt(A, y1[i - 1], K)
    y2[i] = y2[i - 1] + dt * dy_dt(A, y2[i - 1], K)
    t[i] = t[i - 1] + dt

fig = plt.figure(figsize=(15,12))
# now let's plot this and compare to the true solution
plt.plot(t, y1, t, y2)

# define a grid and compute direction at each point
# get axis limits
ymax = plt.ylim()[1]
xmax = plt.xlim()[1]
nb_points = 30

x = np.linspace(0, xmax, nb_points)
y = np.linspace(0, ymax, nb_points)

X, Y = np.meshgrid(x, y)

U = 1
V = dy_dt(A, Y, K)

# to synchronize the scale of grid between 'solution plot, y1, y2' and 'direction fields'
# otherwise, the direction of arrows do not look to be tangential to the solution plot.
V = V * xmax / ymax

M = np.sqrt(U**2+V**2)
# Avoid zero division errors - for example,
# in case we hit the stability point!
M[M == 0] = 1.

# Normalize each arrow by the growth rate's norm
U /= M
V /= M

plt.title('Direction fields : Click on a point to show a specific trajectory')
plt.quiver(X, Y, U, V, M, pivot='mid', cmap=plt.cm.jet)

def onclick(event):
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d', ix, iy)

    earlierNumSteps = int(ix / dt)
    laterNumSteps = int((xmax - ix)/dt)

    print(earlierNumSteps, laterNumSteps)

    start_t = int(ix / dt)
    y_progress = np.zeros(totNumSteps)
    y_progress[start_t] = iy
    print(start_t+1, laterNumSteps)

    for i in np.arange(start_t+1, int(xmax/dt)):
        y_progress[i] = y_progress[i - 1] + dt * dy_dt(A, y_progress[i - 1], K)
        print(ymax)
        if y_progress[i] > ymax:
            y_progress[i] = ymax
        t[i] = t[i - 1] + dt

    for i in np.arange(1, earlierNumSteps):
        y_progress[start_t - i] = y_progress[start_t - i + 1] - dt * dy_dt(A, y_progress[start_t - i +1], K)
        print(ymax)
        if y_progress[start_t - i] > ymax:
            y_progress[start_t - i] = ymax
        t[start_t - i] = t[start_t - i +1] - dt

    plt.plot(t, y_progress)

    # Update Plot
    fig.canvas.draw()

    return True
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()