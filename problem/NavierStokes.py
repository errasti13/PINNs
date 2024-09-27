import numpy as np
import tensorflow as tf

class SteadyNavierStokes2D:

    def __init__(self, nu=0.01):
        self.nu = nu  # Kinematic viscosity (Reynolds number dependent)
        self.problemTag = "SteadyNavierStokes"

    def getBoundaryCondition(self, N0, x_min, x_max, y_min, y_max, sampling_method='uniform'):
        if sampling_method == 'random':
            # Random sampling of boundary points
            xBc_left = np.full((N0, 1), x_min, dtype=np.float32)
            yBc_left = np.random.rand(N0, 1) * (y_max - y_min) + y_min
            
            xBc_right = np.full((N0, 1), x_max, dtype=np.float32)
            yBc_right = np.random.rand(N0, 1) * (y_max - y_min) + y_min

            yBc_bottom = np.full((N0, 1), y_min, dtype=np.float32)
            xBc_bottom = np.random.rand(N0, 1) * (x_max - x_min) + x_min

            yBc_top = np.full((N0, 1), y_max, dtype=np.float32)
            xBc_top = np.random.rand(N0, 1) * (x_max - x_min) + x_min
        elif sampling_method == 'uniform':
            # Uniform grid of boundary points
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

        # Boundary conditions for u, v (velocity components) and pressure p
        uBc_left = np.zeros_like(xBc_left, dtype=np.float32)  # No-slip condition on left boundary
        vBc_left = np.zeros_like(xBc_left, dtype=np.float32)

        uBc_right = np.zeros_like(xBc_right, dtype=np.float32)  # No-slip condition on right boundary
        vBc_right = np.zeros_like(xBc_right, dtype=np.float32)

        uBc_bottom = np.zeros_like(yBc_bottom, dtype=np.float32)  # No-slip condition at bottom
        vBc_bottom = np.zeros_like(yBc_bottom, dtype=np.float32)

        uBc_top = np.ones_like(yBc_top, dtype=np.float32)  # Inlet condition at top (u = 1)
        vBc_top = np.zeros_like(yBc_top, dtype=np.float32)

        return xBc_left, yBc_left, uBc_left, vBc_left, xBc_right, yBc_right, uBc_right, vBc_right, xBc_bottom, yBc_bottom, uBc_bottom, vBc_bottom, xBc_top, yBc_top, uBc_top, vBc_top

    def generate_data(self, x_range, y_range, N0=100, Nf=10000, sampling_method='random'):
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]

        # Boundary data (u, v on boundaries)
        xBc_left, yBc_left, uBc_left, vBc_left, xBc_right, yBc_right, uBc_right, vBc_right, \
        xBc_bottom, yBc_bottom, uBc_bottom, vBc_bottom, xBc_top, yBc_top, uBc_top, vBc_top = self.getBoundaryCondition(N0, x_min, x_max, y_min, y_max, sampling_method)

        # Collocation points (internal points for solving PDE)
        if sampling_method == 'random':
            x_f = (np.random.rand(Nf, 1) * (x_max - x_min) + x_min).astype(np.float32)
            y_f = (np.random.rand(Nf, 1) * (y_max - y_min) + y_min).astype(np.float32)
        elif sampling_method == 'uniform':
            x_f = np.linspace(x_min, x_max, Nf)[:, None].astype(np.float32)
            y_f = np.linspace(y_min, y_max, Nf)[:, None].astype(np.float32)

        return x_f, y_f, xBc_left, yBc_left, uBc_left, vBc_left, xBc_right, yBc_right, uBc_right, vBc_right, \
               xBc_bottom, yBc_bottom, uBc_bottom, vBc_bottom, xBc_top, yBc_top, uBc_top, vBc_top
    
    def loss_function(self, model, data):
        # Unpack the data
        x_f, y_f, xBc_left, yBc_left, uBc_left, vBc_left, xBc_right, yBc_right, uBc_right, vBc_right, \
        xBc_bottom, yBc_bottom, uBc_bottom, vBc_bottom, xBc_top, yBc_top, uBc_top, vBc_top = data

        # Convert boundary data to tensors
        x_f = tf.convert_to_tensor(x_f, dtype=tf.float32)
        y_f = tf.convert_to_tensor(y_f, dtype=tf.float32)
        
        # Convert boundary conditions into tensors
        uBc_left = tf.convert_to_tensor(uBc_left, dtype=tf.float32)
        vBc_left = tf.convert_to_tensor(vBc_left, dtype=tf.float32)
        uBc_right = tf.convert_to_tensor(uBc_right, dtype=tf.float32)
        vBc_right = tf.convert_to_tensor(vBc_right, dtype=tf.float32)
        uBc_bottom = tf.convert_to_tensor(uBc_bottom, dtype=tf.float32)
        vBc_bottom = tf.convert_to_tensor(vBc_bottom, dtype=tf.float32)
        uBc_top = tf.convert_to_tensor(uBc_top, dtype=tf.float32)
        vBc_top = tf.convert_to_tensor(vBc_top, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x_f, y_f])

            # Predict u, v, p at collocation points
            uvp_pred = model(tf.concat([x_f, y_f], axis=1))
            u_pred = uvp_pred[:, 0]
            v_pred = uvp_pred[:, 1]
            p_pred = uvp_pred[:, 2]

            # Compute derivatives using automatic differentiation
            u_x = tape.gradient(u_pred, x_f)
            u_y = tape.gradient(u_pred, y_f)
            v_x = tape.gradient(v_pred, x_f)
            v_y = tape.gradient(v_pred, y_f)
            p_x = tape.gradient(p_pred, x_f)
            p_y = tape.gradient(p_pred, y_f)

            # Second-order derivatives
            u_xx = tape.gradient(u_x, x_f)
            u_yy = tape.gradient(u_y, y_f)
            v_xx = tape.gradient(v_x, x_f)
            v_yy = tape.gradient(v_y, y_f)

        # Incompressibility condition (continuity equation)
        continuity = u_x + v_y

        # X-momentum equation
        momentum_u = u_pred * u_x + v_pred * u_y + p_x - self.nu * (u_xx + u_yy)

        # Y-momentum equation
        momentum_v = u_pred * v_x + v_pred * v_y + p_y - self.nu * (v_xx + v_yy)

        # Collocation losses (PDE residuals)
        f_loss_u = tf.reduce_mean(tf.square(momentum_u))
        f_loss_v = tf.reduce_mean(tf.square(momentum_v))
        continuity_loss = tf.reduce_mean(tf.square(continuity))

        # Boundary condition losses
        u_left_pred = model(tf.concat([tf.cast(xBc_left, dtype=tf.float32), 
                                    tf.cast(yBc_left, dtype=tf.float32)], axis=1))[:, 0]
        v_left_pred = model(tf.concat([tf.cast(xBc_left, dtype=tf.float32), 
                                    tf.cast(yBc_left, dtype=tf.float32)], axis=1))[:, 1]

        u_right_pred = model(tf.concat([tf.cast(xBc_right, dtype=tf.float32), 
                                        tf.cast(yBc_right, dtype=tf.float32)], axis=1))[:, 0]
        v_right_pred = model(tf.concat([tf.cast(xBc_right, dtype=tf.float32), 
                                        tf.cast(yBc_right, dtype=tf.float32)], axis=1))[:, 1]

        # Calculate the L2 loss for boundary conditions
        uBc_loss_left = tf.reduce_mean(tf.square(u_left_pred - uBc_left))
        vBc_loss_left = tf.reduce_mean(tf.square(v_left_pred - vBc_left))
        uBc_loss_right = tf.reduce_mean(tf.square(u_right_pred - uBc_right))
        vBc_loss_right = tf.reduce_mean(tf.square(v_right_pred - vBc_right))
        
        # Similarly for the top and bottom boundary conditions...
        
        # Total loss
        total_loss = (f_loss_u + f_loss_v + continuity_loss +
                    uBc_loss_left + vBc_loss_left + uBc_loss_right + vBc_loss_right)
        
        return total_loss


    def predict(self, pinn, x_range, y_range, Nx=256, Ny=256):
        x_pred = np.linspace(x_range[0], x_range[1], Nx)[:, None].astype(np.float32)
        y_pred = np.linspace(y_range[0], y_range[1], Ny)[:, None].astype(np.float32)
        X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

        # Predict velocity (u, v) and pressure (p) using the trained model
        predictions = pinn.predict(np.hstack((X_pred.flatten()[:, None], Y_pred.flatten()[:, None])))

        # Unpack directly after checking shape
        uPred, vPred, pPred = predictions[:, 0], predictions[:, 1], predictions[:, 2]

        return uPred, vPred, pPred, X_pred, Y_pred

