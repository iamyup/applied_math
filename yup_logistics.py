# Assignment2 - first task
import numpy as np
import matplotlib.pylab as plt


# logistic equation
def dp_dt(P, A=0.08, K=1000):
    return (A * P * (1 - P/K))


def calc_ft(f0_increase, f0_decrease):
    # default parameter
    A = 0.08
    K = 1000

    # ode.integrate in below
    # ft_increase = integrate.odeint(dp_dt, f0_increase, t, args=(A, K,))
    # ft_decrease = integrate.odeint(dp_dt, f0_decrease, t, args=(A, K,))

    # use EULER's METHOD
    # step size
    dt = 0.01
    # total number of integration steps
    totNumSteps = int(100 / dt)

    # define arrays
    ft_increase = np.zeros(totNumSteps)
    ft_decrease = np.zeros(totNumSteps)
    t = np.zeros(totNumSteps)

    # initial function value for two cases
    ft_increase[0] = f0_increase
    ft_decrease[0] = f0_decrease
    t[0] = 0
    for i in np.arange(1, totNumSteps):
        ft_increase[i] = ft_increase[i - 1] + dt * dp_dt(ft_increase[i - 1], A, K)
        ft_decrease[i] = ft_decrease[i - 1] + dt * dp_dt(ft_decrease[i - 1], A, K)
        t[i] = t[i - 1] + dt
    return t, ft_increase, ft_decrease


def direction_field():
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
    V = dp_dt(Y, A, K)

    # to synchronize the scale of grid between solution plot ft_increase, ft_derease and direction fields
    # otherwise, the direction of arrows do not look to be tangential to the solution plot.
    V = V * xmax / ymax

    # norm of logistics
    M = np.sqrt(U ** 2 + V ** 2, dtype=np.float64)
    # avoid zero division errors
    M[M == 0] = 1.

    # normalize each arrow
    U /= M
    V /= M

    # plot the direction field for the range of parameters
    # define a grid and compute direction at each point
    Q = plt.quiver(X, Y, U, V, M, pivot='mid', cmap=plt.cm.jet)


# default parameter
A = 0.08
K = 1000

# initial function value for two cases
f0_increase = 10
f0_decrease = 1500

# solve ft
t, ft_increase, ft_decrease = calc_ft(f0_increase, f0_decrease)

# plot all possible solutions
fig = plt.figure()
plt.plot(t, ft_increase, t, ft_decrease)

# draw direction field
direction_field()


# processing for user click
# poll this location and show the evaluation from that point (towards earlier times and towards later times!).
# redraw based on user click
def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))

    # re-assign init value
    if event.ydata > K:
        re_f0_decrease = event.ydata
        re_f0_increase = f0_increase
    else:
        re_f0_increase = event.ydata
        re_f0_decrease = f0_decrease

    # solve ft using re-assigned value
    re_t, re_ft_increase, re_ft_decrease = calc_ft(re_f0_increase, re_f0_decrease)

    # Update Plot
    # clear screen
    plt.clf()
    # redraw plot
    plt.plot(re_t, re_ft_increase, re_t, re_ft_decrease)
    # re-draw direction field
    direction_field()

    # re-draw carrying capacity & grid
    plt.plot([0, 100], [1000, 1000], 'k--')
    plt.grid()
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)

# plot carrying capacity & grid
plt.plot([0,100],[1000,1000], 'k--')
plt.grid()
plt.show()

