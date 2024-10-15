import numpy as np
import tensorflow as tf

class HeatEquation2D:

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.problemTag = "HeatEquation"

    def getBoundaryCondition(self, N0, x_min, x_max, y_min, y_max, sampling_method='uniform'):
        boundaries = {
            'left': {'x': None, 'y': None, 'u': None},
            'right': {'x': None, 'y': None, 'u': None},
            'bottom': {'x': None, 'y': None, 'u': None},
            'top': {'x': None, 'y': None, 'u': None},
        }

        if sampling_method == 'random':
            # Random sampling of boundary points
            boundaries['left']['x'] = np.full((N0, 1), x_min, dtype=np.float32)
            boundaries['left']['y'] = np.random.rand(N0, 1) * (y_max - y_min) + y_min

            boundaries['right']['x'] = np.full((N0, 1), x_max, dtype=np.float32)
            boundaries['right']['y'] = np.random.rand(N0, 1) * (y_max - y_min) + y_min

            boundaries['bottom']['y'] = np.full((N0, 1), y_min, dtype=np.float32)
            boundaries['bottom']['x'] = np.random.rand(N0, 1) * (x_max - x_min) + x_min

            boundaries['top']['y'] = np.full((N0, 1), y_max, dtype=np.float32)
            boundaries['top']['x'] = np.random.rand(N0, 1) * (x_max - x_min) + x_min

        elif sampling_method == 'uniform':
            # Uniform grid of boundary points
            yBc = np.linspace(y_min, y_max, N0)[:, None].astype(np.float32)
            xBc = np.linspace(x_min, x_max, N0)[:, None].astype(np.float32)

            boundaries['left']['x'] = np.full_like(yBc, x_min, dtype=np.float32)
            boundaries['left']['y'] = yBc

            boundaries['right']['x'] = np.full_like(yBc, x_max, dtype=np.float32)
            boundaries['right']['y'] = yBc

            boundaries['bottom']['y'] = np.full_like(xBc, y_min, dtype=np.float32)
            boundaries['bottom']['x'] = xBc

            boundaries['top']['y'] = np.full_like(xBc, y_max, dtype=np.float32)
            boundaries['top']['x'] = xBc
        else:
            raise ValueError("sampling_method should be 'random' or 'uniform'")
        
        boundaries['left']['u']    = tf.zeros_like(boundaries['left']['x'], dtype=np.float32)
        boundaries['right']['u']   = tf.zeros_like(boundaries['right']['x'], dtype=np.float32)
        boundaries['top']['u']     = tf.zeros_like(boundaries['top']['x'], dtype=np.float32)
        boundaries['bottom']['u']  = tf.ones_like(boundaries['bottom']['x'], dtype=np.float32)

        return boundaries
    
    def generate_data(self, x_range, y_range, N0=100, Nf=10000, sampling_method='random'):
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]

        boundaries = self.getBoundaryCondition(N0, x_min, x_max, y_min, y_max, sampling_method)

        x_f, y_f = [], []
        while len(x_f) < Nf:
            x_candidate = (np.random.rand(1) * (x_max - x_min) + x_min).astype(np.float32)
            y_candidate = (np.random.rand(1) * (y_max - y_min) + y_min).astype(np.float32)
            
            x_f.append(x_candidate)
            y_f.append(y_candidate)

        x_f = np.array(x_f, dtype=np.float32).reshape(-1, 1)
        y_f = np.array(y_f, dtype=np.float32).reshape(-1, 1)

        return x_f, y_f, boundaries
    
    def imposeBoundaryCondition(self, uBc):
        def convert_if_not_none(tensor):
            return tf.convert_to_tensor(tensor, dtype=tf.float32) if tensor is not None else None

        uBc = convert_if_not_none(uBc)

        return uBc

    def computeBoundaryLoss(self, model, xBc, yBc, uBc):
        def compute_loss(bc):
            if bc is not None:
                pred = model(tf.concat([tf.cast(xBc, dtype=tf.float32), tf.cast(yBc, dtype=tf.float32)], axis=1))[:, 0]
                return tf.reduce_mean(tf.square(pred - bc))
            else:
                return tf.constant(0.0)

        uBc_loss = compute_loss(uBc)

        return uBc_loss

    def loss_function(self, model, data):
        x_f, y_f, boundaries = data

        total_loss = 0

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x_f, y_f])

            u_pred = model(tf.concat([x_f, y_f], axis=1))
            u_pred = u_pred[:, 0]

            u_x = tape.gradient(u_pred, x_f)
            u_y = tape.gradient(u_pred, y_f)

            u_xx = tape.gradient(u_x, x_f)
            u_yy = tape.gradient(u_y, y_f)

        # Calculate the internal loss (residual loss) and reduce to a scalar
        f = u_xx + u_yy
        eqLoss = tf.reduce_mean(tf.square(f))  # Reduce to scalar
        
        total_loss += eqLoss  # Add internal loss to total_loss

        # Compute boundary losses
        for boundary_key, boundary_data in boundaries.items():
            xBc = boundary_data['x']
            yBc = boundary_data['y']
            uBc = boundary_data['u']

            uBc_tensor = self.imposeBoundaryCondition(uBc)
            uBc_loss = self.computeBoundaryLoss(model, xBc, yBc, uBc_tensor)

            total_loss += uBc_loss

        return total_loss


    def predict(self, pinn, x_range, y_range, Nx=100, Ny=100):
        # Prediction grid
        x_pred = np.linspace(x_range[0], x_range[1], Nx)[:, None].astype(np.float32)
        y_pred = np.linspace(y_range[0], y_range[1], Ny)[:, None].astype(np.float32)
        X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

        # Predict solution using the trained PINN model
        uPred = pinn.model.predict(np.hstack((X_pred.flatten()[:, None], Y_pred.flatten()[:, None])))

        # Numerical solution for comparison
        x_num = np.linspace(x_range[0], x_range[1], Nx)[:, None].astype(np.float32)
        y_num = np.linspace(y_range[0], y_range[1], Ny)[:, None].astype(np.float32)
        X_num, Y_num = np.meshgrid(x_num, y_num)
        uNumeric = self.numericalSolution(x_range, y_range, Nx, Ny)
        
        return uPred, X_pred, Y_pred, uNumeric, X_num, Y_num

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
