import numpy as np
import matplotlib.pyplot as plt
from Kuramoto import Basic_Kuramoto, Dynamic_Kuramoto

'''
How this script is structured:
- First, the simulation parameters are defined.
- Then, helpler functions inclding plot-functions are defined.
    - Functions that end with basic are for the basic model
    - Functions that end with dynamic are for the time-varying model
- Finally, the plot functions are called that one wants to use.
'''

#------------------------------------------------------------------------------------------
# Define simulation parameters

N = 50  # Number of oscillators
F = 10 # Central natural frequency

spread = 0.5
omega = np.linspace(F-spread,F+spread,N) # Natural frequencies

theta0 = np.random.uniform(0, 2*np.pi, N)  # Initial phases
T = 100  # Total time
dt = 0.01  # Time step

K_values = [0.1, 1, 5]  # Different coupling strengths to simulate for basic Kuramoto model
K_range = np.linspace(0.0001, 5, 40) # Range of coupling strengths for simulation of steady state order parameter

# Parameters for time-varying coupling strength (for definitions look at generate_K function below):
t = np.arange(0, T, dt)  # Time array for time-varying coupling strength
phase_offsets = [0, 3, np.random.uniform(0, 2 * np.pi)]
gamma = 0.7  # DC offset
mu = 0.4  # amplitude
g = 1.0 # frequency
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

# Plot-function definitions

def plot_order_parameter_basic(N=N, omega=omega, theta0=theta0, T=T, dt=dt, K_values=K_values):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))
    fig.suptitle(f'Time Evolution of the Order Parameter R for Different K Values with N = {N} oscillators')

    for ax, K in zip(axes, K_values):
        kuramoto = Basic_Kuramoto(N, K, omega, theta0, T, dt)
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
        kuramoto = Basic_Kuramoto(N, K, omega, theta0, T, dt)
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
        
        kuramoto = Dynamic_Kuramoto(N, K, omega, theta0, T, dt)
        t, theta = kuramoto.simulate()
        
        # Calculate the order parameter R for each time step
        R = np.array([kuramoto.order_parameter(theta_i) for theta_i in theta])
        
        ax.plot(t, R, lw=1)
        ax.set_title(f'freq = 1.0, offset = {psi:.2f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Order Parameter R')

    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------
    
# Call the plot functions that you want to use
    
# plot_order_parameter_basic()
# plot_steady_order_basic()
plot_timevarying_K_dynamic()