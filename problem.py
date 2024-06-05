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

    def loss_function(self, model, x_f, t_f, u0, x0, t0):
        x_f = tf.convert_to_tensor(x_f, dtype=tf.float32)
        t_f = tf.convert_to_tensor(t_f, dtype=tf.float32)
        x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
        t0 = tf.convert_to_tensor(t0, dtype=tf.float32)
        u0 = tf.convert_to_tensor(u0, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x_f, t_f, x0, t0])
            
            u_pred = model(tf.concat([x_f, t_f], axis=1))
            u0_pred = model(tf.concat([x0, t0], axis=1))
            
            u_x = tape.gradient(u_pred, x_f)
            u_t = tape.gradient(u_pred, t_f)
            u_xx = tape.gradient(u_x, x_f)
            
        f = u_t + u_pred * u_x - self.nu * u_xx
        u0_loss = tf.reduce_mean(tf.square(u0_pred - u0))
        f_loss = tf.reduce_mean(tf.square(f))
        
        return u0_loss + f_loss
