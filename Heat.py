import numpy as np
import tensorflow as tf

class HeatEquation2D:

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.problemTag = "HeatEquation"

    def getBoundaryCondition(self, N0, x_min, x_max, y_min, y_max, sampling_method='uniform'):
        if sampling_method == 'random':
            xBc_left = np.full((N0, 1), x_min, dtype=np.float32)
            yBc_left = (np.random.rand(N0, 1) * (y_max - y_min) + y_min).astype(np.float32)

            xBc_right = np.full((N0, 1), x_max, dtype=np.float32)
            yBc_right = (np.random.rand(N0, 1) * (y_max - y_min) + y_min).astype(np.float32)

            yBc_bottom = np.full((N0, 1), y_min, dtype=np.float32)
            xBc_bottom = (np.random.rand(N0, 1) * (x_max - x_min) + x_min).astype(np.float32)

            yBc_top = np.full((N0, 1), y_max, dtype=np.float32)
            xBc_top = (np.random.rand(N0, 1) * (x_max - x_min) + x_min).astype(np.float32)
        elif sampling_method == 'uniform':
            yBc = np.linspace(y_min, y_max, N0)[:, None].astype(np.float32)
            xBc = np.linspace(x_min, x_max, N0)[:, None].astype(np.float32)

            xBc_left = np.full_like(yBc, x_min, dtype=np.float32)
            yBc_left = yBc

            xBc_right = np.full_like(yBc, x_max, dtype=np.float32)
            yBc_right = yBc

            yBc_bottom = np.full_like(xBc, y_min, dtype=np.float32)
            xBc_bottom = xBc

            yBc_top = np.full_like(xBc, y_max, dtype=np.float32)
            xBc_top = xBc
        else:
            raise ValueError("sampling_method should be 'random' or 'uniform'")

        uBc_left = np.zeros_like(xBc_left, dtype=np.float32)
        uBc_right = np.zeros_like(xBc_right, dtype=np.float32)
        uBc_bottom = np.zeros_like(yBc_bottom, dtype=np.float32)
        uBc_top = np.ones_like(yBc_top, dtype=np.float32)

        return xBc_left, yBc_left, uBc_left, xBc_right, yBc_right, uBc_right, xBc_bottom, yBc_bottom, uBc_bottom, xBc_top, yBc_top, uBc_top

    def generate_data(self, x_range, y_range, N0=100, Nf=10000, sampling_method='uniform'):
        x_min, x_max = x_range
        y_min, y_max = y_range

        xBc_left, yBc_left, uBc_left, xBc_right, yBc_right, uBc_right, xBc_bottom, yBc_bottom, uBc_bottom, xBc_top, yBc_top, uBc_top = self.getBoundaryCondition(N0, x_min, x_max, y_min, y_max, sampling_method)

        x_f = (np.random.rand(Nf, 1) * (x_max - x_min) + x_min).astype(np.float32)
        y_f = (np.random.rand(Nf, 1) * (y_max - y_min) + y_min).astype(np.float32)

        return x_f, y_f, xBc_left, yBc_left, uBc_left, xBc_right, yBc_right, uBc_right, xBc_bottom, yBc_bottom, uBc_bottom, xBc_top, yBc_top, uBc_top

    def loss_function(self, model, data):
        x_f, y_f, xBc_left, yBc_left, uBc_left, xBc_right, yBc_right, uBc_right, xBc_bottom, yBc_bottom, uBc_bottom, xBc_top, yBc_top, uBc_top = data

        x_f = tf.convert_to_tensor(x_f, dtype=tf.float32)
        y_f = tf.convert_to_tensor(y_f, dtype=tf.float32)

        xBc_left = tf.convert_to_tensor(xBc_left, dtype=tf.float32)
        yBc_left = tf.convert_to_tensor(yBc_left, dtype=tf.float32)
        uBc_left = tf.convert_to_tensor(uBc_left, dtype=tf.float32)

        xBc_right = tf.convert_to_tensor(xBc_right, dtype=tf.float32)
        yBc_right = tf.convert_to_tensor(yBc_right, dtype=tf.float32)
        uBc_right = tf.convert_to_tensor(uBc_right, dtype=tf.float32)

        xBc_bottom = tf.convert_to_tensor(xBc_bottom, dtype=tf.float32)
        yBc_bottom = tf.convert_to_tensor(yBc_bottom, dtype=tf.float32)
        uBc_bottom = tf.convert_to_tensor(uBc_bottom, dtype=tf.float32)

        xBc_top = tf.convert_to_tensor(xBc_top, dtype=tf.float32)
        yBc_top = tf.convert_to_tensor(yBc_top, dtype=tf.float32)
        uBc_top = tf.convert_to_tensor(uBc_top, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x_f, y_f, xBc_left, yBc_left, xBc_right, yBc_right, xBc_bottom, yBc_bottom, xBc_top, yBc_top])

            u_pred = model(tf.concat([x_f, y_f], axis=1))
            uBc_left_pred = model(tf.concat([xBc_left, yBc_left], axis=1))
            uBc_right_pred = model(tf.concat([xBc_right, yBc_right], axis=1))
            uBc_bottom_pred = model(tf.concat([xBc_bottom, yBc_bottom], axis=1))
            uBc_top_pred = model(tf.concat([xBc_top, yBc_top], axis=1))

            u_x = tape.gradient(u_pred, x_f)
            u_y = tape.gradient(u_pred, y_f)
            u_xx = tape.gradient(u_x, x_f)
            u_yy = tape.gradient(u_y, y_f)

        f = u_xx + u_yy
        uBc_left_loss = tf.reduce_mean(tf.square(uBc_left_pred - uBc_left))
        uBc_right_loss = tf.reduce_mean(tf.square(uBc_right_pred - uBc_right))
        uBc_bottom_loss = tf.reduce_mean(tf.square(uBc_bottom_pred - uBc_bottom))
        uBc_top_loss = tf.reduce_mean(tf.square(uBc_top_pred - uBc_top))
        f_loss = tf.reduce_mean(tf.square(f))

        return uBc_left_loss + uBc_right_loss + uBc_bottom_loss + uBc_top_loss + f_loss


    def numericalSolution(self, xRange, yRange, Nx, Ny):
        x_min, x_max = xRange[0], xRange[1]
        y_min, y_max = yRange[0], yRange[1]

        # Initialize the temperature field
        u = np.zeros((Nx, Ny))

        # Set boundary conditions (example: u = 0 at boundaries)
        u[:, 0] = 0  # u = 0 at x_min
        u[:, -1] = 0  # u = 0 at x_max
        u[0, :] = 0  # u = 0 at y_min
        u[-1, :] = 1  # u = 0 at y_max

        # Tolerance and iteration parameters for the iterative solver
        tol = 1e-6
        max_iter = 10000
        iter_count = 0

        # Iterative solver (Gauss-Seidel method)
        while iter_count < max_iter:
            u_old = u[1:-1, 1:-1].copy()
            
            # Update the solution for interior points
            u[1:-1, 1:-1] = 0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])
            
            # Compute the error efficiently
            error = np.linalg.norm(u[1:-1, 1:-1] - u_old, ord=2)

            iter_count += 1

            if iter_count % 1000 == 0:
                print(f"Iteration {iter_count}, Error: {error}")

            if error < tol:
                break

        if iter_count == max_iter:
            print("Warning: Maximum number of iterations reached. Solution may not have converged.")

        return u
