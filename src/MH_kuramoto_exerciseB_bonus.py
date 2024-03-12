# Import necessary libraries
import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint

N = 2  #setting limited number of oscillators for sake of simplicity 
F = 10  #same as in task 1
frequency_spread = 3  #same as in task 1
scale = 1  #pure guess that 1 suitable for scale of Gaussian noise
sampling_rate = 200  #same as in task 1
time = 5 #ALINA MODIFICATION 11.3.24

#generating oscillator frequencies
frequencies = np.linspace(F - frequency_spread, F + frequency_spread, N) # with F = 10 and frequency_spread = 3 -> [7. 13.]

#t need to be as variable as well, because odeint function passes the current time t to this differential equation kuramoto_model at each integration step
def kuramoto_model(theta, t, coupling):
    #intializing dtheta_dt
    dtheta_dt = np.zeros_like(theta)
    for i in range(N):
        omega = frequencies[i] # ALINA MODIFICATION 11.3.24: omega is frequency for this node from the list
        #assuring that (theta_j - theta_i) is not calcualted for same oscillators i.e. j =! i
        phase_diffs = np.array([theta[j] - theta[i] for j in range(N) if j != i]) # ALINA MODIFICATION 11.3.24: difference in theta with other oscillators 
        dtheta_dt[i] = omega + (coupling/N) * np.sum(np.sin(phase_diffs)) + np.random.normal(0, scale) # ALINA MODIFICATION 11.3.24
    return dtheta_dt

#simulating the Kuramoto model with different coupling values
coupling_values = [0.1, 2] #ALINA MODIFICATION 11.3.24
plt.figure(figsize=(10, 6))

for coupling in coupling_values:
    #random initial phase for each oscillator
    initial_phase = np.random.uniform(0, 2 * np.pi, N)

    #time array
    times = np.linspace(0, time, time * sampling_rate)

    #odeint for solving ordinary differential equations. atol and rtol specifies absolute and relative tolerance for solutions
    theta = odeint(kuramoto_model, initial_phase, times, args=(coupling,), atol=1e-8, rtol=1e-6) #ALINA MODIFICATION 11.3.24

    #plotting results
    oscillations = np.exp(1j*theta) # ALINA MODIFICATION 11.3.24: phase in complex values 
    plt.plot(times, np.angle(oscillations[:, 0]), linestyle="--", label=f"Oscillator 1, coupling {coupling}") # ALINA MODIFICATION 11.3.24: np.angle to show phase
    plt.plot(times, np.angle(oscillations[:, 1]), label=f"Oscillator 2, coupling {coupling}") # ALINA MODIFICATION 11.3.24

plt.xlabel('Time')
plt.ylabel('Phase')
plt.title('Kuramoto model simulation of two oscillators with different coupling values')
plt.legend()
plt.show()