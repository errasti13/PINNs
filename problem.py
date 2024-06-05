import numpy as np
import tensorflow as tf

class BurgersEquation:
    def __init__(self, nu=0.01 / np.pi):
        self.nu = nu

    def getInitialSolution(self, N0, x_min, x_max, t_min, sampling_method):
        if sampling_method == 'random':
            x0 = (np.random.rand(N0, 1) * (x_max - x_min) + x_min).astype(np.float32)  # Initial condition: x in [x_min, x_max]
        elif sampling_method == 'uniform':
            x0 = np.linspace(x_min, x_max, N0)[:, None].astype(np.float32)  # Uniform sampling: x in [x_min, x_max]
        
        t0 = np.full((N0, 1), t_min, dtype=np.float32)  # Initial condition: t = t_min
        u0 = -np.sin(2 * np.pi * x0 / (x_max - x_min)).astype(np.float32)  # Initial velocity

        return x0, t0, u0
    

    def getBoundaryCondition(self, N0, t_min, t_max, x, sampling_method='uniform'):
        # Generate time coordinates based on the sampling method
        if sampling_method == 'random':
            tBc = (np.random.rand(N0, 1) * (t_max - t_min) + t_min).astype(np.float32)  # Random sampling in [t_min, t_max]
        elif sampling_method == 'uniform':
            tBc = np.linspace(t_min, t_max, N0)[:, None].astype(np.float32)  # Uniform sampling in [t_min, t_max]
        else:
            raise ValueError("sampling_method should be 'random' or 'uniform'")

        xBc = np.full((N0, 1), x, dtype=np.float32)  # Boundary x-coordinate is constant

        uBc = np.zeros((N0, 1), dtype=np.float32)  # Initial velocity/condition

        return xBc, tBc, uBc

    def generate_data(self, x_range, t_range, N0=100, Nf=10000, sampling_method='uniform'):
        x_min, x_max = x_range
        t_min, t_max = t_range

        x0, t0, u0 = self.getInitialSolution(N0, x_min, x_max, t_min, sampling_method)

        xBc0, tBc0, uBc0 = self.getBoundaryCondition(N0, t_min, t_max, x_min, sampling_method)
        xBc1, tBc1, uBc1 = self.getBoundaryCondition(N0, t_min, t_max, x_max, sampling_method)

        x_f = (np.random.rand(Nf, 1) * (x_max - x_min) + x_min).astype(np.float32)  # Collocation points: x in [x_min, x_max]
        t_f = (np.random.rand(Nf, 1) * (t_max - t_min) + t_min).astype(np.float32)  # Collocation points: t in [t_min, t_max]

        return x_f, t_f, u0, x0, t0, xBc0, tBc0, uBc0, xBc1, tBc1, uBc1 

    def loss_function(self, model, data):

        x_f, t_f, u0, x0, t0, xBc0, tBc0, uBc0, xBc1, tBc1, uBc1 = data

        x_f = tf.convert_to_tensor(x_f, dtype=tf.float32)
        t_f = tf.convert_to_tensor(t_f, dtype=tf.float32)
        x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
        t0 = tf.convert_to_tensor(t0, dtype=tf.float32)
        u0 = tf.convert_to_tensor(u0, dtype=tf.float32)

        xBc0 = tf.convert_to_tensor(xBc0, dtype=tf.float32)
        tBc0 = tf.convert_to_tensor(tBc0, dtype=tf.float32)
        uBc0 = tf.convert_to_tensor(uBc0, dtype=tf.float32)

        xBc1 = tf.convert_to_tensor(xBc1, dtype=tf.float32)
        tBc1 = tf.convert_to_tensor(tBc1, dtype=tf.float32)
        uBc1 = tf.convert_to_tensor(uBc1, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x_f, t_f, x0, t0])
            
            u_pred = model(tf.concat([x_f, t_f], axis=1))
            u0_pred = model(tf.concat([x0, t0], axis=1))

            uBc0_pred = model(tf.concat([xBc0, tBc0], axis = 1))
            uBc1_pred = model(tf.concat([xBc1, tBc1], axis = 1))
            
            u_x = tape.gradient(u_pred, x_f)
            u_t = tape.gradient(u_pred, t_f)
            u_xx = tape.gradient(u_x, x_f)
            
        f = u_t + u_pred * u_x - self.nu * u_xx
        u0_loss = tf.reduce_mean(tf.square(u0_pred - u0))
        uBc0_loss = tf.reduce_mean(tf.square(uBc0_pred - uBc0))
        uBc1_loss = tf.reduce_mean(tf.square(uBc1_pred - uBc1))
        f_loss = tf.reduce_mean(tf.square(f))
        
        return u0_loss+ uBc0_loss + uBc1_loss + f_loss
    
    def numericalSolution(self, xRange, tRange, Nx, Nt):
        # Parameters
        nu = self.nu  # Viscosity
        x_min, x_max = xRange[0], xRange[1]
        t_min, t_max = tRange[0], tRange[1]

        # Discretization
        dx = (x_max - x_min) / (Nx - 1)
        dt = (t_max - t_min) / Nt

        # Stability criterion
        alpha = nu * dt / dx**2
        if alpha > 0.5:
            print("Warning: The solution may be unstable. Consider reducing dt or increasing dx.")

        # Initial condition
        x = np.linspace(x_min, x_max, Nx)
        u_initial = -np.sin(np.pi * x)

        # Initialize u
        u = u_initial.copy()
        u_new = np.zeros_like(u)

        # Time-stepping loop
        for n in range(1, Nt + 1):
            t = n * dt
            for i in range(1, Nx - 1):
                u_new[i] = u[i] - dt * u[i] * (u[i] - u[i - 1]) / dx + alpha * (u[i + 1] - 2 * u[i] + u[i - 1])
            
            # Boundary conditions
            u_new[0] = 0.0
            u_new[-1] = 0.0
            
            # Update u
            u = u_new.copy()

        return u

