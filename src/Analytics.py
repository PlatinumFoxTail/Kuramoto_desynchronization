import numpy as np
import matplotlib.pyplot as plt
from Kuramoto import Basic_Kuramoto, Dynamic_Kuramoto

'''
Considerations:
- The plots are not looking like the ones in the paper. I don't know what is wrong, requires further investigation. I will continue in the beginning of next week.
'''

# Here we test the basic kuramoto model by plotting the time evolution of the order parameter for different coupling strengths.

# Parameters
N = 4  # Number of oscillators
F = 10 # Central natural frequency
scale = 1 # Scale of the Lorentzian distribution

# We generate natural frequencies from a Lorentzian distribution
while True:
    omega = F + scale * np.random.standard_cauchy(N)
    # Truncate the frequencies to remove very large values
    omega = omega[(omega > F - 3*scale) & (omega < F + 3*scale)]
    if len(omega) >= N:
        break
omega = omega[:N]  # Take only the first N values

theta0 = np.random.uniform(0, 2*np.pi, N)  # Initial phases
T = 100  # Total time
dt = 0.01  # Time step

K_values = [0.01, 0.1, 1]  # Different coupling strengths to simulate

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))
fig.suptitle('Time Evolution of the Order Parameter R for Different K Values')

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


# Now we test the dynamic kuramoto model by plotting the time evolution of the order parameter for different time-varying coupling strengths.

# We use the same parameters as for the basic model

# We generate time-varying coupling strengths like was done in the Cumin. et al. paper:
t = np.arange(0, T, dt)

# Compute time-varying K values outside the class (example with a simple sine wave)
# Function to generate K values
def generate_K(t, gamma, mu, g, psi):
    return gamma + mu * np.sin(2 * np.pi * g * t + psi)

# Plot similar to row 3 of Fig 7 in Cummin et al. paper
# Create a subplot for each phase offset
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
phase_offsets = [0, 3, np.random.uniform(0, 2 * np.pi)]

for ax, psi in zip(axes, phase_offsets):
    # Generate K values with the given phase offset
    K = generate_K(t, gamma=0.7, mu=0.4, g=1.0, psi=psi)
    
    kuramoto = Dynamic_Kuramoto(N, K, omega, theta0, T, dt)
    t, theta = kuramoto.simulate()
    
    # Calculate the order parameter R for each time step
    R = np.abs(np.sum(np.exp(1j * theta), axis=1) / N)
    
    ax.plot(t, R, lw=1)
    ax.set_title(f'K(t) freq = 1.0, K(t) offset = {psi:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Order Parameter R')

plt.tight_layout()
plt.show()