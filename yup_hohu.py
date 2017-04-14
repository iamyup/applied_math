# Assignment 3: solves the Hodgkin-Huxley equations for the parameters shown in the ODE.pptx lecture.
import sys
import numpy as np

# should support several modes
def iext(t):
    return t # ??

def ode(X, t):
    V, m, n, h = X

    # slide 66
    gk = 1 # ??
    Ek = 1 # ??
    n = (1-np.exp(t)) # ??
    gna = 1 # ??
    h = 1 # ??
    Ena = 1 # ??
    gL = 1 # ??
    EL = 1 # ??
    C = 1 # ??
    dV_dt = (iext(t) - gk * (n**4) * (V - Ek) - gna * (m**3) * h * (V - Ena) - gL * (V - EL)) / C

    # slides 64, 65
    # alphan =
    # alpham =
    # alphah =
    # betan =
    # betam =
    # betah =
    #
    # # slide 66
    # dm_dt =
    # dn_dt =
    # dh_dt =

    return (...)


# necessary parameters for the ODEs
# Maximal conductances as np.array
# a = potassium(K), 2=sodium(NA), 3=leakage
g = np.array([36, 120, 0.3])

# Battery voltages as np.array
# a=n-particle, 2=m-particle, 3=h-particle
E = np.array([-12, 115, 1 - .613])

# time for integration (in ms)
# note that we start at negative times to allow the membrane to relax
# its dynamics, before we start plotting at t = 0
time = np.linspace(-30, 50, 8000)

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
            I = np.array([1e-5, 1.5, 2, 2.5, 3, 4, 5, 10, 15, 20, 50])
        elif arg is "2":
            # two close-by stimulations that do not result in two spikes for I_ext=15 stimulation given from 10 - 12 ms and 19 - 21 ms
            I = np.array([15.0])
        elif arg is "3":
            # two close-by stimulations that do result in two spikes for I_ext=15 stimulation continuously from 10 - 12 ms and t0 - t0+2 ms
            # [FOUND OUT THE **CLOSEST** TIMEPOINT t0 THAT GIVES TWO SPIKES THROUGH EXPERIMENTATION - I DO NOT NEED TO SEE THE CODE TO DO THIS, THE VALUE t0 IS ENOUGH IN THE SCRIPT!!]
            I = np.array([15.0])
        elif arg is "4":
            # Long stimuulation with several current injections stimulation continuously from 10 - 40 ms
            # [THIS MODE SHOULD ALSO BE USED TO FIND OUT THE MAXIMUM FIRING FREQUENCY!]
            # The current injections should be taken from the following array
            I = np.array([10, 15, 20, 50])
        elif arg is "5":
            # No stimulation, but plot the membrane potential from -30 - 0 ms
            # [THIS MODE SHOULD BE USED TO DESCRIBE THE BEHAVIOR OF THE NEURON FOR TIMES BEFORE 0]
            I = np.array([0])
        else:
            # No stimulation
            I = np.array([0])

if __name__ == '__main__':
    main()