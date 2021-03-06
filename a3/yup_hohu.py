# Assignment 3: solves the Hodgkin-Huxley equations for the parameters shown in the ODE.pptx lecture.
import sys
import numpy as np
from matplotlib import pyplot as plt

# ordinary differential equation of Hodgkin-Huxley equation
def ode(i_ext, X, t):
    V, n, m, h = X

    # slide 66
    gk = g[0]
    Ek = E[0]
    i_k = gk * (n**4) * (V - Ek)

    gna = g[1]
    Ena = E[1]
    i_na = gna * (m**3) * h * (V - Ena)

    gl = g[2]
    El = E[2]
    i_l = gl * (V - El)

    C = 1 # ??
    dV_dt = (i_ext(t) - i_k - i_na - i_l) / C

    # slides 64, 65
    alphan = (0.01 * (10 - V)) / (np.exp((10 - V) / 10) - 1)
    betan = 0.125 * np.exp(-V / 80)

    alpham = 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)
    betam = 4.0 * np.exp(-V / 18)

    alphah = 0.07 * np.exp(-V / 20)
    betah = 1 / (np.exp((30 - V) / 10) + 1)

    # slide 66
    dn_dt = alphan * (1 - n) - betan * n
    dm_dt = alpham * (1 - m) - betam * m
    dh_dt = alphah * (1 - h) - betah * h
    return dV_dt, dn_dt, dm_dt, dh_dt


# euler method for differential equation
def euler_method(i_ext, t, dt):
    totNumSteps = len(t)

    vt = np.zeros(totNumSteps)
    nt = np.zeros(totNumSteps)
    mt = np.zeros(totNumSteps)
    ht = np.zeros(totNumSteps)

    vt[0] = s0[0]
    nt[0] = s0[1]
    mt[0] = s0[2]
    ht[0] = s0[3]
    for i in np.arange(1, totNumSteps):
        X = np.array([vt[i-1], nt[i-1], mt[i-1], ht[i-1]])
        dv_dt, dn_dt, dm_dt, dh_dt = ode(i_ext, X, t[i])

        vt[i] = vt[i - 1] + dt * dv_dt
        nt[i] = nt[i - 1] + dt * dn_dt
        mt[i] = mt[i - 1] + dt * dm_dt
        ht[i] = ht[i - 1] + dt * dh_dt

    return vt, nt, mt, ht


# necessary parameters for the ODEs
# Maximal conductances as np.array
# a = potassium(K):gk, 2=sodium(NA):gna, 3=leakage:gl
g = np.array([36, 120, 0.3])

# Battery voltages as np.array
# a=n-particle:Ek, 2=m-particle:Ena, 3=h-particle:El
E = np.array([-12, 115, 10.613])

# time for integration (in ms)
# note that we start at negative times to allow the membrane to relax
# its dynamics, before we start plotting at t = 0
time = np.linspace(-30, 50, 8001, retstep=True) # we changed step number from 8000 => 8001
#print(time)
# starting values for V, n, m, h
s0 = np.array([-10, 0, 0, 0])


# main function
def main():
    # rename time to t
    t = time[0]
    # rename time step to dt
    dt = time[1]

    # index set for time value is greater than 0
    smaller_0 = np.where(t < 0)
    greater_0 = np.where(t > 0)

    # (1) If the mode only supplies ONE current injection or NO current injection, then the code should draw TWO plots:
    # First plot: the solution V(t) plotted from t=0:end INCLUDING the current injection time points clearly marked (shaded region, box, etc.) using plt.ylim([-20,120])
    # Second plot: the solutions n,m,h plotted from t=0:end INCLUDING the current injection time points clearly marked (shaded region, box, etc.)
    #
    # (2) If the mode supplies SEVERAL current injections, then the code should only draw one plot for each injection:
    # Plot: the solution V(t) plotted from t=0:end INCLUDING the current injection time points clearly marked (shaded region, box, etc.) using plt.ylim([-20,120])inj
    #
    # (3) If the mode is 5, then the output plot should be:
    # Plot: the solution V(t) plotted from t=-30:0 (NO SCALING for ylim!!)
    # process each mode that input by user
    for arg in sys.argv[1:]:
        if arg is "1":
            # One stimulation timepoint with several current injections stimulation, with each stimulation given from 10 - 12 ms
            # The current injections should be taken from the following array
            I = np.array([1e-5, 1.5, 2, 2.5, 3, 4, 5, 10, 15, 20, 50])

            for idx, i in enumerate(I):
                def i_ext(x):
                    return i * (x >= 10) - i * (x > 12)

                V, n, m, h = euler_method(i_ext, t, dt)

                plt.figure(idx, figsize=(10, 12))
                ax1 = plt.subplot(211)
                ax1.plot(t[greater_0], i_ext(t[greater_0]), label='i_Ext : '+str(i)+' (micro_A/cm2)')
                ax1.plot(t[greater_0], V[greater_0], label='V(t) : response to i_Ext')
                plt.ylim([-20, 120])
                ax1.legend(loc=1)

                ax2 = plt.subplot(212)
                ax2.plot(t[greater_0], i_ext(t[greater_0]), label='i_Ext : '+str(i)+' (micro_A/cm2)')
                ax2.plot(t[greater_0], n[greater_0]*100, label='n-gating variable') # multiply 100 for clearance of the graph
                ax2.plot(t[greater_0], m[greater_0]*100, label='m-gating variable') # multiply 100 for clearance of the graph
                ax2.plot(t[greater_0], h[greater_0]*100, label='h-gating variable') # multiply 100 for clearance of the graph
                plt.ylim([-10, 120])
                ax2.legend(loc=1)

        elif arg is "2":
            # two close-by stimulations that do not result in two spikes for I_ext=15 stimulation given from 10 - 12 ms and 19 - 21 ms
            I = np.array([15.0])

            for idx, i in enumerate(I):
                def i_ext(x):
                    return i * (x >= 10) - i * (x > 12) + i * (x >= 19) - i * (x > 21)

                V, n, m, h = euler_method(i_ext, t, dt)

                plt.figure(idx, figsize=(10, 6))
                plt.plot(t[greater_0], i_ext(t[greater_0]), label='Injection(i_Ext) : '+str(i)+' (micro_A/cm2)')
                plt.plot(t[greater_0], V[greater_0], label='V(t) : response to i_Ext')
                plt.ylim([-20, 120])
                plt.legend(loc=1)

        elif arg is "3":
            # two close-by stimulations that do result in two spikes for I_ext=15 stimulation continuously from 10 - 12 ms and t0 - t0+2 ms
            # [FOUND OUT THE **CLOSEST** TIMEPOINT t0 THAT GIVES TWO SPIKES THROUGH EXPERIMENTATION -
            # I DO NOT NEED TO SEE THE CODE TO DO THIS, THE VALUE t0 IS ENOUGH IN THE SCRIPT!!]
            I = np.array([15.0])
            for idx, i in enumerate(I):
                def i_ext(x):
                    t0 = 19.64
                    '''
                    note that it's not 19.66.
                    we set time step numbers as 8001 instead of 8000
                    because if we set steps as 8000, the step size become 0.01000125 not just 0.01
                    it makes difference of 0.02(ms)

                    *** yes, it can be different if we use "integrate.odeint" of scipy instead of euler_method.
                    we did not check yet.
                    '''

                    return i * (x >= 10) - i * (x > 12) + i * (x >= t0) - i * (x > (t0+2))

                V, n, m, h = euler_method(i_ext, t, dt)

                plt.figure(idx, figsize=(10, 6))
                plt.plot(t[greater_0], i_ext(t[greater_0]), label='Injections(i_Ext) at 10(ms) and 19.64(ms)')
                plt.plot(t[greater_0], V[greater_0], label='V(t) for two close-by stimulations')
                plt.ylim([-20, 120])
                plt.legend(loc=1)

        elif arg is "4":
            # Long stimulation with several current injections stimulation continuously from 10 - 40 ms
            # [THIS MODE SHOULD ALSO BE USED TO FIND OUT THE MAXIMUM FIRING FREQUENCY!]
            # The current injections should be taken from the following array

            # result of our analysis
            ''' basically in the greater stimulation(external current), the more spikes are generated
            , which means HIGHER FREQUENCY.
            too much injection over I_ext = 150, however, kills the firing train.

            and it's not so clear about the maximum frequency. it depends on how high spike is a spike!

            for I_ext = 100, it generates almost 5 peaks during 30ms. and its frequency is 167 Hz.
            but its peak is less than 50 mV.
            if the peak must satisfy to be over 50 mV, it generates almost 4 peaks during 30ms. and its frequency is 133 Hz.

            finally, we can say that THE MAXIMUM FIRING FREQUENCY of this neuron is around 150Hz (or 130~170Hz).
            '''
            I = np.array([10, 15, 20, 50])
            #I = np.array([50])
            for idx, i in enumerate(I):
                def i_ext(x):
                    return i * (x >= 10) - i * (x > 40)

                V, n, m, h = euler_method(i_ext, t, dt)

                #plt.figure(idx)
                plt.figure(idx, figsize=(10, 12))
                ax1 = plt.subplot(211)
                ax1.plot(t[greater_0], i_ext(t[greater_0]), label='i_Ext : '+str(i)+' (micro_A/cm2)')
                ax1.plot(t[greater_0], V[greater_0], label='V(t) : response to i_Ext')
                plt.ylim([-20, 120])
                ax1.legend(loc=1)

                ax2 = plt.subplot(212)
                ax2.plot(t[greater_0], i_ext(t[greater_0]), label='i_Ext : '+str(i)+' (micro_A/cm2)')
                ax2.plot(t[greater_0], n[greater_0]*100, label='n-gating variable') # multiply 100 for clearance of the graph
                ax2.plot(t[greater_0], m[greater_0]*100, label='m-gating variable') # multiply 100 for clearance of the graph
                ax2.plot(t[greater_0], h[greater_0]*100, label='h-gating variable') # multiply 100 for clearance of the graph
                plt.ylim([-10, 120])
                ax2.legend(loc=1)

        elif arg is "5":
            # No stimulation, but plot the membrane potential from -30 - 0 ms
            # [THIS MODE SHOULD BE USED TO DESCRIBE THE BEHAVIOR OF THE NEURON FOR TIMES BEFORE 0]
            ''' Although there is no stimulation, a spike is found near t = -24.
            but why? and how come is this possible?
            it comes from equilibrium potentials, i.e E_K, E_Na, E_l.
            For the initial condition, the membrane capacitor is empty(no electric charge).
            So some current flows and the capacitor collects charges until reaching "Equilibrium" with E_K, E_Na, E_l.
            that's why we can see the peak in the early part.
            '''
            # describe the behavior of the membrane for the times before 0
            # Although no stimulation, there is a spike near t's -24.

            I = np.array([0])
            for idx, i in enumerate(I):
                def i_ext(x):
                    return i*x

                V, n, m, h = euler_method(i_ext, t, dt)

                #plt.figure(idx)
                plt.figure(idx, figsize=(10, 12))
                ax1 = plt.subplot(211)
                ax1.plot(t[smaller_0], i_ext(t[smaller_0]), label='i_Ext : '+str(i)+' (micro_A/cm2)')
                ax1.plot(t[smaller_0], V[smaller_0], label='V(t) : response for equilibrium')
                plt.ylim([-20, 120])
                ax1.legend(loc=1)

                ax2 = plt.subplot(212)
                ax2.plot(t[smaller_0], i_ext(t[smaller_0]), label='i_Ext : '+str(i)+' (micro_A/cm2)')
                ax2.plot(t[smaller_0], n[smaller_0]*100, label='n-gating variable') # multiply 100 for clearance of the graph
                ax2.plot(t[smaller_0], m[smaller_0]*100, label='m-gating variable') # multiply 100 for clearance of the graph
                ax2.plot(t[smaller_0], h[smaller_0]*100, label='h-gating variable') # multiply 100 for clearance of the graph
                plt.ylim([-10, 120])
                ax2.legend(loc=1)
        else:
            # No stimulation
            I = np.array([0])

    plt.show()

if __name__ == '__main__':
    main()
