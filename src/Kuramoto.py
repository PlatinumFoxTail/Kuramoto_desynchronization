import numpy as np
from scipy.integrate import odeint, solve_ivp

'''
Open questions:
- Do we want to include the option to have different coupling strengths for different pairs of oscillators?
'''


class Basic_Kuramoto:
    def __init__(self, N, K, omega, theta0, T, dt):
        '''
        Simple fully connected Kuramoto model with constant natural frequencies and constant coupling strength.

        Parameters:
            N:      number of oscillators
            K:      coupling strength
            omega:  array of natural frequencies for each oscillator
            theta0: array of initial phases for each oscillator
            T:      total time
            dt:     time step
        '''
        self.N = N
        self.K = K
        self.omega = omega
        self.theta0 = theta0
        self.T = T
        self.dt = dt

    def kuramoto_dynamics(self, theta, t):
        '''
        Differential equation describing Kuramoto model dynamics. Used for solving the ODE with odeint.

        Parameters:
            theta:  array of phases at time t
            t:      time

        Returns:
            dtheta_dt: array of phase derivatives
        '''
        dtheta_dt = self.omega + (self.K / self.N) * np.sum(np.sin(np.subtract.outer(theta, theta)), axis=0)
        return dtheta_dt
    
    def simulate(self):
        '''
        Simulate the Kuramoto model using odeint differential equation solver.

        Returns:
            t:      array of time points
            theta:  array of phase trajectories
        '''
        t = np.arange(0, self.T, self.dt)
        theta = odeint(self.kuramoto_dynamics, self.theta0, t)
    
        return t, theta
    
    def order_parameter(self, theta):
        '''
        Calculate the order parameter for a given set of phases.

        Parameters:
            theta:  array of phases

        Returns:
            r:      order parameter
        '''
        r = np.abs(np.sum(np.exp(1j * theta))/self.N)
        return r



class Dynamic_Kuramoto:
    def __init__(self, N, K, omega, theta0, T, dt):
        '''
        Dynamic Kuramoto model with time-varying coupling streangths.

        Parameters:
            N:      number of oscillators
            K:      coupling strengths as a function of time [Array of shape (T/dt, 1)]
            omega:  array of natural frequencies for each oscillator
            theta0: array of initial phases for each oscillator
            T:      total time
            dt:     time step
        '''
        self.N = N
        self.K = K
        self.omega = omega
        self.theta0 = theta0
        self.T = T
        self.dt = dt

    def kuramoto_dynamics(self, theta, t):
        '''
        Differential equation describing Kuramoto model dynamics. Used for solving the ODE with odeint.

        Parameters:
            theta:  array of phases at time t
            t:      time

        Returns:
            dtheta_dt: array of phase derivatives
        '''
        # Find the index in the K_values array that corresponds to the current time step
        K_index = min(int(t / self.dt), len(self.K) - 1)
        dtheta_dt = self.omega + (self.K[K_index] / self.N) * np.sum(np.sin(np.subtract.outer(theta, theta)), axis=0)
        return dtheta_dt
    
    def simulate(self):
        '''
        Simulate the Kuramoto model using odeint differential equation solver.

        Returns:
            t:      array of time points
            theta:  array of phase trajectories
        '''
        t = np.arange(0, self.T, self.dt)
        theta = odeint(self.kuramoto_dynamics, self.theta0, t)
        return t, theta
    
    def order_parameter(self, theta):
        '''
        Calculate the order parameter for a given set of phases.

        Parameters:
            theta:  array of phases

        Returns:
            r:      order parameter
        '''
        r = np.abs(np.sum(np.exp(1j * theta))/self.N)
        return r