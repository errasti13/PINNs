import numpy as np
import tensorflow as tf

class SteadyNavierStokes2D:
    
    def __init__(self, nu=0.01):
        self.nu = nu  # Kinematic viscosity (Reynolds number dependent)

    def getBoundaryCondition(self, N0, x_min, x_max, y_min, y_max, sampling_method='uniform'):
        # Boundary conditions will be implemented by each specific problem
        raise NotImplementedError("Boundary condition method must be implemented in a subclass.")
    
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
    
    def imposeBoundaryCondition(self, uBc, vBc):
        if uBc is not None and vBc is not None:
            uBc = tf.convert_to_tensor(uBc, dtype=tf.float32)
            vBc = tf.convert_to_tensor(vBc, dtype=tf.float32)

        return uBc, vBc
    
    def computeBoundaryLoss(self, model, xBc, yBc, uBc, vBc):
        if uBc is not None and vBc is not None:
            uPred = model(tf.concat([tf.cast(xBc, dtype=tf.float32), tf.cast(yBc, dtype=tf.float32)], axis=1))[:, 0]
            vPred = model(tf.concat([tf.cast(xBc, dtype=tf.float32), tf.cast(yBc, dtype=tf.float32)], axis=1))[:, 1]

            return tf.reduce_mean(tf.square(uPred - uBc)), tf.reduce_mean(tf.square(vPred - vBc))
        
        else:
            return tf.constant(0.0)
        
    def loss_function(self, model, data):
        # Unpack the data
        x_f, y_f, xBc_left, yBc_left, uBc_left, vBc_left, xBc_right, yBc_right, uBc_right, vBc_right, \
        xBc_bottom, yBc_bottom, uBc_bottom, vBc_bottom, xBc_top, yBc_top, uBc_top, vBc_top = data

        # Convert boundary data to tensors
        uBc_left, vBc_left = self.imposeBoundaryCondition(uBc_left, vBc_left)
        uBc_right, vBc_right = self.imposeBoundaryCondition(uBc_right, vBc_right)
        uBc_top, vBc_top = self.imposeBoundaryCondition(uBc_top, vBc_top)
        uBc_bottom, vBc_bottom = self.imposeBoundaryCondition(uBc_bottom, vBc_bottom)

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

        uBc_loss_left, vBc_loss_left = self.computeBoundaryLoss(model, xBc_left, yBc_left, uBc_left, vBc_left)
        uBc_loss_right, vBc_loss_right = self.computeBoundaryLoss(model, xBc_right, yBc_right, uBc_right, vBc_right)
        uBc_loss_top, vBc_loss_top = self.computeBoundaryLoss(model, xBc_top, yBc_top, uBc_top, vBc_top)
        uBc_loss_bottom, vBc_loss_bottom = self.computeBoundaryLoss(model, xBc_bottom, yBc_bottom, uBc_bottom, vBc_bottom)

        # Total loss
        total_loss = (f_loss_u + f_loss_v + continuity_loss + 
                        uBc_loss_left + vBc_loss_left +
                    uBc_loss_right + vBc_loss_right +
                    uBc_loss_top + vBc_loss_top +
                    uBc_loss_bottom + vBc_loss_bottom)

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

# Example problem class, like LidDrivenCavity
class LidDrivenCavity(SteadyNavierStokes2D):
    
    def __init__(self, nu=0.01):
        super().__init__(nu)
        self.problemTag = "LidDrivenCavity"

        return
    
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
    

class ChannelFlow(SteadyNavierStokes2D):

    def getBoundaryCondition(self, N0, x_min, x_max, y_min, y_max, 
                             sampling_method='uniform', enforce_outlet=True):
        if sampling_method == 'random':
            xBc_left = np.full((N0, 1), x_min, dtype=np.float32)
            yBc_left = np.random.rand(N0, 1) * (y_max - y_min) + y_min

            xBc_right = np.full((N0, 1), x_max, dtype=np.float32)
            yBc_right = np.random.rand(N0, 1) * (y_max - y_min) + y_min

            yBc_bottom = np.full((N0, 1), y_min, dtype=np.float32)
            xBc_bottom = np.random.rand(N0, 1) * (x_max - x_min) + x_min

            yBc_top = np.full((N0, 1), y_max, dtype=np.float32)
            xBc_top = np.random.rand(N0, 1) * (x_max - x_min) + x_min
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

        # Boundary conditions for u, v
        uBc_left = np.ones_like(yBc_left, dtype=np.float32) * (1 - (yBc_left / y_max) ** 2)  # Parabolic inlet
        vBc_left = np.zeros_like(yBc_left, dtype=np.float32)  # No vertical velocity at inlet

        if enforce_outlet:
            # Outlet: no boundary conditions (do-nothing approach)
            uBc_right = np.zeros_like(xBc_right, dtype=np.float32)
            vBc_right = np.zeros_like(xBc_right, dtype=np.float32)
        else:
            # No boundary condition at the outlet (do nothing)
            uBc_right = None
            vBc_right = None

        # No-slip conditions for top and bottom
        uBc_bottom = np.zeros_like(xBc_bottom, dtype=np.float32)
        vBc_bottom = np.zeros_like(xBc_bottom, dtype=np.float32)
        
        uBc_top = np.ones_like(xBc_top, dtype=np.float32)
        vBc_top = np.zeros_like(xBc_top, dtype=np.float32)

        return (xBc_left, yBc_left, uBc_left, vBc_left,
                xBc_right, yBc_right, uBc_right, vBc_right,
                xBc_bottom, yBc_bottom, uBc_bottom, vBc_bottom,
                xBc_top, yBc_top, uBc_top, vBc_top)

