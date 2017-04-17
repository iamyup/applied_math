# Assignment 3: solves the Hodgkin-Huxley equations for the parameters shown in the ODE.pptx lecture.
import sys
import numpy as np

# should support several modes
def iext(t):
    return t # ??


def ode(X, t):
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
    dV_dt = (iext(t) - i_k - i_na - i_l) / C

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
time = np.linspace(-30, 50, 8000, retstep=True)

# starting values for V, n, m, h
s0 = np.array([-10, 0, 0, 0])


# (1) Mode 3 below requires you to experiment with the ODE a little bit in order to find out the minimum time that a stimulation needs to have in order to elicit two separate spikes.

# (2) Mode 5, insert a COMMENT into the script that describes the behavior of the membrane for the times before 0

# (3) know what Maximum firing frequency of this neuron is
# Please experiment with the ODE in Mode 4 and insert the result of your anaysis as a COMMENT into the script.


def main():
    for arg in sys.argv[1:]:
        print(arg)

        if arg is "1":
            # One stimulation timepoint with several current injections stimulation, with each stimulation given from 10 - 12 ms
            # The current injections should be taken from the following array
            I_ext = np.array([1e-5, 1.5, 2, 2.5, 3, 4, 5, 10, 15, 20, 50])
        elif arg is "2":
            # two close-by stimulations that do not result in two spikes for I_ext=15 stimulation given from 10 - 12 ms and 19 - 21 ms
            I_ext = np.array([15.0])
        elif arg is "3":
            # two close-by stimulations that do result in two spikes for I_ext=15 stimulation continuously from 10 - 12 ms and t0 - t0+2 ms
            # [FOUND OUT THE **CLOSEST** TIMEPOINT t0 THAT GIVES TWO SPIKES THROUGH EXPERIMENTATION - I DO NOT NEED TO SEE THE CODE TO DO THIS, THE VALUE t0 IS ENOUGH IN THE SCRIPT!!]
            I_ext = np.array([15.0])
        elif arg is "4":
            # Long stimuulation with several current injections stimulation continuously from 10 - 40 ms
            # [THIS MODE SHOULD ALSO BE USED TO FIND OUT THE MAXIMUM FIRING FREQUENCY!]
            # The current injections should be taken from the following array
            I_ext = np.array([10, 15, 20, 50])
        elif arg is "5":
            # No stimulation, but plot the membrane potential from -30 - 0 ms
            # [THIS MODE SHOULD BE USED TO DESCRIBE THE BEHAVIOR OF THE NEURON FOR TIMES BEFORE 0]
            I_ext = np.array([0])
        else:
            # No stimulation
            I_ext = np.array([0])

    dt = time[1]
    totNumSteps = len(time[0])

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
        dv_dt, dn_dt, dm_dt, dh_dt = ode(X, time[0][i])

        vt[i] = vt[i - 1] + dt * dv_dt
        nt[i] = nt[i - 1] + dt * dn_dt
        mt[i] = mt[i - 1] + dt * dm_dt
        ht[i] = ht[i - 1] + dt * dh_dt

if __name__ == '__main__':
    main()