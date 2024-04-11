import numpy as np
import matplotlib.pyplot as plt
from Kuramoto import Kuramoto, Time_varying_coupling_Kuramoto

'''
How this script is structured:
- First, the simulation parameters are defined.
- Then, helpler functions inclding plot-functions are defined.
    - Functions that end with basic are for the basic model
    - Functions that end with dynamic are for the time-varying model
- Finally, the plot functions are called that one wants to use.
'''

'''
Conciderations from the Cummin et al. paper:
- Each plot has many simulation runs shown.
- A sinusoidal time-varying function has been chosen with the initial phase of each sine-wave 
  being randomised for each oscillator.
- The frequency g is set vary beteen 0 and 1, where 0 is represents low freqeuncies and 1 high frequencies.
- The coupling strength was varied from 0.3 and 1.1 which was identified as the region of interest.
    - For larger N, we first need to identify the region of interest and then choose parameters of K(t) accordingly.
- The columns of Fig. 7 in Cummin et al. paper, (moving from left to right) represent phase offsets psi (0, π, random).
'''

#------------------------------------------------------------------------------------------
# Define simulation parameters

N = 4  # Number of oscillators
F = 10 # Central natural frequency

spread = 0.5
omega = np.linspace(F-spread,F+spread,N) # Natural frequencies

theta0 = np.random.uniform(0, 2*np.pi, N)  # Initial phases
T = 100  # Total time
dt = 0.01  # Time step

K_values = [0.1, 1, 5]  # Different coupling strengths to simulate for basic Kuramoto model
K_range = np.linspace(0.0001, 5, 40) # Range of coupling strengths for simulation of steady state order parameter
m=10 # Number of simulation runs for each phase offset

# Parameters for time-varying coupling strength (for definitions look at generate_K function below):

#---------------------------------------------
# Default parameters as defined in Cummin et al. paper, works for 4 oscillators
gamma = 0.7  # DC offset
mu = 0.4  # amplitude
g = 1.0 # frequency
g_values = [0.01, 0.1, 1.0]  # Different K frequencies to plot similar as Cummin et al. paper
phase_offsets = [0, np.pi, np.random.uniform(0, 2*np.pi)]  # Phase offsets


# For larger N define new ones according to the region of interest
#---------------------------------------------

t = np.arange(0, T, dt)  # Time array for time-varying coupling strength

#-------------------------------------------------------------------------------------------

# Function to calculate the time-varying coupling strength based on Cummin et al. paper
def generate_K(t, gamma, mu, g, psi):
    """
    Calculate the time-varying coupling strength K_ij(t) for the Kuramoto model.

    Parameters:
    - t: time
    - gamma: DC offset
    - mu: amplitude of the time-varying component
    - g: frequency of the time-varying component
    - psi: phase offset of the time-varying component

    Returns:
    - K_ij(t): time-varying coupling strength at time t
    """
    return gamma + mu * np.sin(2 * np.pi * g * t + psi)

# Function to generate natural frequencies for oscillators to make similar plots to what Cummin et al. paper shows

# Define the distribution g(ω) for sampling the natural frequencies (from Cummin et al. paper)
def g_omega(omega):
    if abs(omega) < 1:
        return (1 - omega**2) / (np.pi - 2) * (1 + omega**2)
    else:
        return 0

# Sample natural frequencies from the distribution
def sample_natural_frequencies(N):
    # This function will return N sampled natural frequencies using the provided distribution g(ω).
    # Since the distribution is piecewise, we will use rejection sampling to sample from it.
    natural_frequencies = []
    while len(natural_frequencies) < N:
        # Sample ω from a uniform distribution that covers the range of g(ω)
        omega = np.random.uniform(-2, 2)  # Adjust the range if necessary
        # Calculate the maximum value of g(ω) for rejection sampling, which occurs at ω=0
        max_g_omega = g_omega(0)
        # Generate a random height
        h = np.random.uniform(0, max_g_omega)
        # Check if this height is below the curve of g(ω)
        if h <= g_omega(omega):
            natural_frequencies.append(omega)
    return natural_frequencies

# Plot-function definitions

def plot_order_parameter_basic(N=N, omega=omega, theta0=theta0, T=T, dt=dt, K_values=K_values):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))
    fig.suptitle(f'Time Evolution of the Order Parameter R for Different K Values with N = {N} oscillators')

    for ax, K in zip(axes, K_values):
        kuramoto = Kuramoto(N, K, omega, theta0, T, dt)
        t, theta = kuramoto.simulate()
        R = np.array([kuramoto.order_parameter(theta_i) for theta_i in theta])
        ax.plot(t, R)
        ax.set_title(f'K = {K}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Order Parameter R')

    plt.tight_layout()
    plt.show()



def plot_steady_order_basic(N=N, omega=omega, theta0=theta0, T=T, dt=dt, K_range=K_range):
    # Array to hold the average order parameter for each K
    avg_R_values = []

    # Simulate for each K value and calculate average R
    for K in K_range:
        kuramoto = Kuramoto(N, K, omega, theta0, T, dt)
        t, theta = kuramoto.simulate()
        R = np.array([kuramoto.order_parameter(theta_i) for theta_i in theta])

        # Average R over the last 50 units of time
        avg_R = np.mean(R[-int(50/dt):])
        avg_R_values.append(avg_R)

    # Plot the average R as a function of K
    plt.figure(figsize=(10, 5))
    plt.plot(K_range, avg_R_values, 'o-')
    plt.title('Average Order Parameter R over the last 50 units of time for different K values')
    plt.xlabel('Coupling strength K')
    plt.ylabel('Average Order Parameter R')
    plt.grid(True)
    plt.show()

def plot_timevarying_K_dynamic(N=N, omega=omega, theta0=theta0, T=T, dt=dt, t=t, gamma=gamma, mu=mu, g=g, phase_offsets=phase_offsets):
    # Plot similar to row 3 of Fig 7 in Cummin et al. paper
    # Create a subplot for each phase offset
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f'Time Evolution of the Order Parameter R for Different Phase Offsets with N = {N} oscillators')

    for ax, psi in zip(axes, phase_offsets):
        # Generate K values with the given phase offset
        K = generate_K(t, gamma=gamma, mu=mu, g=g, psi=psi)
        
        kuramoto = Time_varying_coupling_Kuramoto(N, K, omega, theta0, T, dt)
        t, theta = kuramoto.simulate()
        
        # Calculate the order parameter R for each time step
        R = np.array([kuramoto.order_parameter(theta_i) for theta_i in theta])
        
        ax.plot(t, R, lw=1)
        ax.set_title(f'freq = 1.0, offset = {psi:.2f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Order Parameter R')

    plt.tight_layout()
    plt.show()

def plot_timevarying_K_dynamic_m_runs(N=N, T=T, dt=dt, t=t, gamma=gamma, mu=mu, g_values=g_values, phase_offsets=phase_offsets, m=m):
    # Create a subplot for each phase offset
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    fig.suptitle(f'Time Evolution of the Order Parameter R for Different Phase Offsets with N = {N} oscillators, across {m} runs', fontsize=12)

    for i, g in enumerate(g_values):
        for j, psi in enumerate(phase_offsets):
            ax = axes[i, j]
            # Loop over m simulation runs
            for _ in range(m):
                # Generate random initial phases for each oscillator
                theta0 = np.random.uniform(0, 2*np.pi, N)
                # Generate natural frequencies for each oscillator 
                omega = sample_natural_frequencies(N)
                # Generate K values with the given phase offset for each run
                K = generate_K(t, gamma=gamma, mu=mu, g=g, psi=psi)
                
                # Initialize the Kuramoto model with the generated K values
                kuramoto = Time_varying_coupling_Kuramoto(N, K, omega, theta0, T, dt)
                t, theta = kuramoto.simulate()
                
                # Calculate the order parameter R for each time step
                R = np.array([kuramoto.order_parameter(theta_i) for theta_i in theta])
                
                # Plot the order parameter R for this run
                ax.plot(t, R, lw=1, alpha=0.5)  # Use a lower alpha to make individual runs distinguishable

            ax.set_title(f'freq = {g}, offset = {psi:.2f}', fontsize=10)
            ax.set_xlabel('Time', fontsize=9)
            ax.set_ylabel('Order Parameter R', fontsize=9)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # plt.tight_layout()
    plt.show()

def plot_g_omega_distribution(omega):
    g_omega_values = [g_omega(omega_val) for omega_val in omega]
    
    plt.figure(figsize=(8, 6))
    plt.plot(omega, g_omega_values, label='g(omega)')
    plt.title('Probability density as a function of omega')
    plt.xlabel('Natural frequency omega')
    plt.ylabel('Probability density g(omega)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_Kij(t, gamma, mu, g_ij, phase_offset):
    # Calculating the time-varying coupling strength
    K = generate_K(t, gamma=gamma, mu=mu, g=g_ij, psi=phase_offset)
    
    plt.figure(figsize=(8, 6))
    plt.plot(t, K)
    plt.title(f'Time-varying coupling strength K_ij(t) for frequency offset {g_ij} and phase offset {phase_offset}')
    plt.xlabel('Time')
    plt.ylabel('Coupling strength K_ij(t)')
    plt.grid(True)
    plt.show()

#-------------------------------------------------------------------------------------------
    
# Call the plot functions that you want to use
    
# plot_order_parameter_basic()
# plot_steady_order_basic()
# plot_timevarying_K_dynamic()
# plot_timevarying_K_dynamic_m_runs()
# plot_g_omega_distribution(omega)
plot_Kij(t, gamma=0.7, mu=0.4, g_ij=1, phase_offset=3.14)
