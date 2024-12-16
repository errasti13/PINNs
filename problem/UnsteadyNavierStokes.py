import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from PIL import Image
import os

class UnsteadyNavierStokes2D:
    
    def __init__(self,):
        return

    def getBoundaryCondition(self, N0, x_min, x_max, y_min, y_max, t_min, t_max, sampling_method='uniform'):
        raise NotImplementedError("Boundary condition method must be implemented in a subclass.")
    
    def generate_data(self, x_range, y_range, t_range, N0=10000, Nf=1000, sampling_method='random'):
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]
        t_min, t_max = t_range[0], t_range[1]

        boundaries = self.getBoundaryCondition(N0, x_min, x_max, y_min, y_max, sampling_method)

        if sampling_method == 'random':
            x_f = (np.random.rand(Nf, 1) * (x_max - x_min) + x_min).astype(np.float32)
            y_f = (np.random.rand(Nf, 1) * (y_max - y_min) + y_min).astype(np.float32)
            t_f = (np.random.rand(N0, 1) * (t_max - t_min) + t_min).astype(np.float32)
        elif sampling_method == 'uniform':
            x_f = np.linspace(x_min, x_max, Nf)[:, None].astype(np.float32)
            y_f = np.linspace(y_min, y_max, Nf)[:, None].astype(np.float32)
            t_f = np.linspace(t_min, t_max, N0)[:, None].astype(np.float32)

        return x_f, y_f, t_f, boundaries
    
    def imposeBoundaryCondition(self, uBc, vBc, pBc):
        def convert_if_not_none(tensor):
            return tf.convert_to_tensor(tensor, dtype=tf.float32) if tensor is not None else None

        uBc = convert_if_not_none(uBc)
        vBc = convert_if_not_none(vBc)
        pBc = convert_if_not_none(pBc)

        return uBc, vBc, pBc

    
    def computeBoundaryLoss(self, model, xBc, yBc, uBc, vBc, pBc):
        def compute_loss(bc, idx):
            if bc is not None:
                pred = model(tf.concat([tf.cast(xBc, dtype=tf.float32), tf.cast(yBc, dtype=tf.float32)], axis=1))[:, idx]
                return tf.reduce_mean(tf.square(pred - bc))
            else:
                return tf.constant(0.0)

        uBc_loss = compute_loss(uBc, 0)
        vBc_loss = compute_loss(vBc, 1)
        pBc_loss = compute_loss(pBc, 2)

        return uBc_loss, vBc_loss, pBc_loss
        
    
    def loss_function(self, model, data):
        x_f, y_f, t_f, boundaries = data

        total_loss = 0

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x_f, y_f, t_f])

            uvp_pred = model(tf.concat([x_f, y_f, t_f], axis=1))
            u_pred = uvp_pred[:, 0]
            v_pred = uvp_pred[:, 1]
            p_pred = uvp_pred[:, 2]

            u_x = tape.gradient(u_pred, x_f)
            u_y = tape.gradient(u_pred, y_f)
            v_x = tape.gradient(v_pred, x_f)
            v_y = tape.gradient(v_pred, y_f)
            p_x = tape.gradient(p_pred, x_f)
            p_y = tape.gradient(p_pred, y_f)

            u_t = tape.gradient(u_pred, t_f)
            v_t = tape.gradient(v_pred, t_f)

            u_xx = tape.gradient(u_x, x_f)
            u_yy = tape.gradient(u_y, y_f)
            v_xx = tape.gradient(v_x, x_f)
            v_yy = tape.gradient(v_y, y_f)

        continuity = u_x + v_y

        momentum_u = u_t + u_pred * u_x + v_pred * u_y + p_x - self.nu * (u_xx + u_yy)

        momentum_v = v_t + u_pred * v_x + v_pred * v_y + p_y - self.nu * (v_xx + v_yy)

        f_loss_u = tf.reduce_mean(tf.square(momentum_u))
        f_loss_v = tf.reduce_mean(tf.square(momentum_v))
        continuity_loss = tf.reduce_mean(tf.square(continuity))

        for boundary_name, boundary in boundaries.items():
            
            uvp_boundary = model(tf.concat([boundary['x'], boundary['y'], boundary['t']], axis=1))
            u_boundary = uvp_boundary[:, 0]
            v_boundary = uvp_boundary[:, 1]
            p_boundary = uvp_boundary[:, 2]

            if boundary['u'] is not None:
                loss_u = tf.reduce_mean(tf.square(u_boundary - boundary['u']))
                total_loss += loss_u

            if boundary['v'] is not None:
                loss_v = tf.reduce_mean(tf.square(v_boundary - boundary['v']))
                total_loss += loss_v

            if boundary['p'] is not None:
                loss_p = tf.reduce_mean(tf.square(p_boundary - boundary['p']))
                total_loss += loss_p

        total_loss += continuity_loss + f_loss_u + f_loss_v

        return total_loss


    def predict(self, pinn, x_range, y_range, Nx=256, Ny=256):
        x_pred = np.linspace(x_range[0], x_range[1], Nx)[:, None].astype(np.float32)
        y_pred = np.linspace(y_range[0], y_range[1], Ny)[:, None].astype(np.float32)
        X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

        predictions = pinn.predict(np.hstack((X_pred.flatten()[:, None], Y_pred.flatten()[:, None])))

        uPred, vPred, pPred = predictions[:, 0], predictions[:, 1], predictions[:, 2]

        return uPred, vPred, pPred, X_pred, Y_pred
    
class UnsteadyFlowOverAirfoil(UnsteadyNavierStokes2D):
    
    def __init__(self, c=1, AoA=0.0, uInlet=1.0, Re = 100):
        super().__init__()
        self.problemTag = "FlowOverAirfoil"
        self.c = c  
        self.AoA = AoA * np.pi / 180 
        self.uInlet = uInlet
        self.nu = c * uInlet / Re
        self.generate_airfoil_coords()

    def generate_airfoil_coords(self, N=100, thickness=0.12):
        x = np.linspace(0, self.c, N)
        x_normalized = x / self.c
        y = 5 * thickness * (0.2969 * np.sqrt(x_normalized) 
                                - 0.1260 * x_normalized 
                                - 0.3516 * x_normalized**2 
                                + 0.2843 * x_normalized**3 
                                - 0.1015 * x_normalized**4)
        self.xAirfoil = x.reshape(-1, 1)
        self.yAirfoil = y.reshape(-1, 1)

        return

    def is_point_inside_airfoil(self, x, y):
        if x < 0 or x > self.c:
            return False

        idx = (np.abs(self.xAirfoil - x)).argmin()

        y_upper = self.yAirfoil[idx]
        y_lower = -y_upper  # Assuming symmetry for the NACA0012 airfoil

        if y_lower <= y <= y_upper:
            return True

        return False

    def getBoundaryCondition(self, N0, x_min, x_max, y_min, y_max, t_min, t_max, sampling_method='uniform', xLE = 0.0, yLE = 0.0):
        boundaries = {
            'left': {'x': None, 'y': None, 't': None, 'v': None, 'p': None},
            'right':{'x': None, 'y': None, 't': None, 'v': None, 'p': None},
            'bottom': {'x': None, 'y': None, 't': None, 'v': None, 'p': None},
            'top': {'x': None, 'y': None, 't': None, 'v': None, 'p': None},
            'airfoil': {'x': None, 'y': None, 't': None, 'v': None, 'p': None}, 
            'initial': {'x': None, 'y': None, 't': None, 'v': None, 'p': None}
        }
    
        boundaries['airfoil']['y'] = self.yAirfoil.astype(np.float32)
        boundaries['airfoil']['x'] = self.xAirfoil.astype(np.float32)


        if sampling_method == 'random':
            # Random sampling of boundary points
            boundaries['left']['x'] = np.full((N0, 1), x_min, dtype=np.float32)
            boundaries['left']['y'] = (np.random.rand(N0, 1) * (y_max - y_min) + y_min).astype(np.float32)
            boundaries['left']['t'] = (np.random.rand(N0, 1) * (t_max - t_min) + t_min).astype(np.float32)

            boundaries['right']['x'] = np.full((N0, 1), x_max, dtype=np.float32)
            boundaries['right']['y'] = (np.random.rand(N0, 1) * (y_max - y_min) + y_min).astype(np.float32)
            boundaries['right']['t'] = (np.random.rand(N0, 1) * (t_max - t_min) + t_min).astype(np.float32)

            boundaries['bottom']['y'] = np.full((N0, 1), y_min, dtype=np.float32)
            boundaries['bottom']['x'] = (np.random.rand(N0, 1) * (x_max - x_min) + x_min).astype(np.float32)
            boundaries['bottom']['t'] = (np.random.rand(N0, 1) * (t_max - t_min) + t_min).astype(np.float32)

            boundaries['top']['y'] = np.full((N0, 1), y_max, dtype=np.float32)
            boundaries['top']['x'] = (np.random.rand(N0, 1) * (x_max - x_min) + x_min).astype(np.float32)
            boundaries['top']['t'] = (np.random.rand(N0, 1) * (t_max - t_min) + t_min).astype(np.float32)

            boundaries['initial']['y'] = (np.random.rand(N0, 1) * (y_max - y_min) + y_min).astype(np.float32)
            boundaries['initial']['x'] = (np.random.rand(N0, 1) * (x_max - x_min) + x_min).astype(np.float32)
            boundaries['initial']['t'] = np.full((N0, 1), t_min, dtype=np.float32).astype(np.float32)

            boundaries['airfoil']['t'] = (np.random.rand(len(self.xAirfoil), 1) * (t_max - t_min) + t_min).astype(np.float32)

        elif sampling_method == 'uniform':
            yBc = np.linspace(y_min, y_max, N0)[:, None].astype(np.float32)
            xBc = np.linspace(x_min, x_max, N0)[:, None].astype(np.float32)
            tBc = np.linspace(t_min, t_max, len(self.xAirfoil))[:, None].astype(np.float32)

            boundaries['left']['x'] = np.full_like(yBc, x_min, dtype=np.float32)
            boundaries['left']['y'] = yBc
            boundaries['left']['t'] = tBc

            boundaries['right']['x'] = np.full_like(yBc, x_max, dtype=np.float32)
            boundaries['right']['y'] = yBc
            boundaries['right']['t'] = tBc

            boundaries['bottom']['y'] = np.full_like(xBc, y_min, dtype=np.float32)
            boundaries['bottom']['x'] = xBc
            boundaries['bottom']['t'] = tBc

            boundaries['top']['y'] = np.full_like(xBc, y_max, dtype=np.float32)
            boundaries['top']['x'] = xBc
            boundaries['top']['t'] = tBc

            boundaries['airfoil']['t'] = tBc

            boundaries['initial']['y'] = yBc
            boundaries['initial']['x'] = xBc
            boundaries['initial']['t'] = np.full_like(xBc, t_min, dtype=np.float32)
        else:
            raise ValueError("sampling_method should be 'random' or 'uniform'")
        
        boundaries['left']['u'] = self.uInlet * np.cos(self.AoA)*tf.ones_like(boundaries['left']['x'], dtype=np.float32)
        boundaries['left']['v'] = self.uInlet * np.sin(self.AoA)*tf.ones_like(boundaries['left']['y'], dtype=np.float32)
        boundaries['left']['p'] = tf.zeros_like(boundaries['right']['x'], dtype=np.float32)

        boundaries['top']['u'] = self.uInlet * np.cos(self.AoA)*tf.ones_like(boundaries['top']['x'], dtype=np.float32)
        boundaries['top']['v'] = self.uInlet * np.sin(self.AoA)*tf.ones_like(boundaries['top']['y'], dtype=np.float32)

        boundaries['right']['u'] = None
        boundaries['right']['v'] = None

        boundaries['bottom']['u'] = self.uInlet * np.cos(self.AoA)*tf.ones_like(boundaries['bottom']['x'], dtype=np.float32)
        boundaries['bottom']['v'] = self.uInlet * np.sin(self.AoA)*tf.ones_like(boundaries['bottom']['y'], dtype=np.float32)

        boundaries['airfoil']['u'] = tf.zeros_like(boundaries['airfoil']['x'], dtype=np.float32)
        boundaries['airfoil']['v'] = tf.zeros_like(boundaries['airfoil']['y'], dtype=np.float32)

        boundaries['initial']['u'] = self.uInlet * np.cos(self.AoA)*tf.ones_like(boundaries['initial']['x'], dtype=np.float32)
        boundaries['initial']['v'] = self.uInlet * np.sin(self.AoA)*tf.ones_like(boundaries['initial']['x'], dtype=np.float32)
        boundaries['initial']['p'] = tf.zeros_like(boundaries['initial']['t'], dtype=np.float32)

        return boundaries
    
    def generate_data(self, x_range, y_range, t_range, N0=100, Nf=10000, sampling_method='random'):
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]
        t_min, t_max = t_range[0], t_range[1]

        boundaries = self.getBoundaryCondition(N0, x_min, x_max, y_min, y_max, t_min, t_max, sampling_method)
        x_f, y_f, t_f = [], [], []

        batch_size = 100  # Adjust for faster sampling in bulk
        while len(x_f) < Nf:
            x_candidates = (np.random.rand(batch_size, 1) * (x_max - x_min) + x_min).astype(np.float32)
            y_candidates = (np.random.rand(batch_size, 1) * (y_max - y_min) + y_min).astype(np.float32)
            t_candidates = (np.random.rand(batch_size, 1) * (t_max - t_min) + t_min).astype(np.float32)

            mask = np.array([not self.is_point_inside_airfoil(x, y) for x, y in zip(x_candidates, y_candidates)], dtype=bool)
            x_f.extend(x_candidates[mask][:Nf - len(x_f)])
            y_f.extend(y_candidates[mask][:Nf - len(y_f)])
            t_f.extend(t_candidates[mask][:Nf - len(t_f)])

        x_f = np.array(x_f, dtype=np.float32).reshape(-1, 1)
        y_f = np.array(y_f, dtype=np.float32).reshape(-1, 1)
        t_f = np.array(t_f, dtype=np.float32).reshape(-1, 1)

        return x_f, y_f, t_f, boundaries
    
    def predict(self, pinn, x_range, y_range, t_range, Nx=256, Ny=1256, Nt=50):
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]
        t_min, t_max = t_range[0], t_range[1]

        time_points = np.linspace(t_min, t_max, Nt)

        uPred_all, vPred_all, pPred_all, xPred_all, yPred_all, tPred_all = [], [], [], [], [], []

        for t in time_points:
            x_pred, y_pred = [], []
            batch_size = 100 
            Nt = Nx * Ny

            while len(x_pred) < Nt:
                x_candidates = (np.random.rand(batch_size, 1) * (x_max - x_min) + x_min).astype(np.float32)
                y_candidates = (np.random.rand(batch_size, 1) * (y_max - y_min) + y_min).astype(np.float32)

                mask = np.array([not self.is_point_inside_airfoil(x, y) for x, y in zip(x_candidates, y_candidates)], dtype=bool)
                x_pred.extend(x_candidates[mask][:Nt - len(x_pred)])
                y_pred.extend(y_candidates[mask][:Nt - len(y_pred)])

            x_pred = np.array(x_pred, dtype=np.float32).reshape(-1, 1)
            y_pred = np.array(y_pred, dtype=np.float32).reshape(-1, 1)
            t_pred = np.full_like(x_pred, t).reshape(-1, 1)

            predictions = pinn.predict(np.hstack((x_pred, y_pred, t_pred)))
            uPred, vPred, pPred = predictions[:, 0], predictions[:, 1], predictions[:, 2]

            uPred_all.append(uPred)
            vPred_all.append(vPred)
            pPred_all.append(pPred)
            xPred_all.append(x_pred[:, 0])
            yPred_all.append(y_pred[:, 0])
            tPred_all.append(t_pred[:, 0])

        uPred_all = np.vstack(uPred_all)
        vPred_all = np.vstack(vPred_all)
        pPred_all = np.vstack(pPred_all)
        xPred_all = np.vstack(xPred_all)
        yPred_all = np.vstack(yPred_all)
        tPred_all = np.vstack(tPred_all)

        return uPred_all, vPred_all, pPred_all, xPred_all, yPred_all, tPred_all

    def plot(self, uPred, vPred, pPred, X_pred, Y_pred, uRange, vRange, pRange):

        plt.figure(figsize=(16, 8))

        # Plot uPred
        plt.subplot(3, 1, 1)
        plt.plot(self.xAirfoil, self.yAirfoil, color='black')
        plt.plot(self.xAirfoil, -self.yAirfoil, color='black')
        plt.scatter(X_pred, Y_pred, c=uPred, vmin = uRange[0], vmax = uRange[1], cmap='jet', s=2)
        plt.xlim(X_pred.min(), X_pred.max())
        plt.ylim(Y_pred.min(), Y_pred.max())
        plt.colorbar()
        plt.title('Predicted U Velocity')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Plot vPred
        plt.subplot(3, 1, 2)
        plt.plot(self.xAirfoil, self.yAirfoil, color='black')
        plt.plot(self.xAirfoil, -self.yAirfoil, color='black')
        plt.scatter(X_pred, Y_pred, c=vPred, vmin = vRange[0], vmax = vRange[1], cmap='jet', s=2)
        plt.xlim(X_pred.min(), X_pred.max())
        plt.ylim(Y_pred.min(), Y_pred.max())
        plt.colorbar()
        plt.title('Predicted V Velocity')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Plot pPred (Pressure)
        plt.subplot(3, 1, 3)
        plt.plot(self.xAirfoil, self.yAirfoil, color='black')
        plt.plot(self.xAirfoil, -self.yAirfoil, color='black')
        plt.scatter(X_pred, Y_pred, c=pPred, vmin = pRange[0], vmax = pRange[1], cmap='jet', s=2)
        plt.xlim(X_pred.min(), X_pred.max())
        plt.ylim(Y_pred.min(), Y_pred.max())
        plt.colorbar()
        plt.title('Predicted Pressure')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.tight_layout()

        return

    def create_gif_from_plots(self, uPred, vPred, pPred, X_pred, Y_pred, T_pred, gif_name="output.gif"):
        from io import BytesIO
        frames = []

        uRange = [uPred.min(), uPred.max()]
        vRange = [vPred.min(), vPred.max()]
        pRange = [pPred.min(), pPred.max()]

        for i in range(len(T_pred[:, 0])):
            # Generate the plot
            self.plot(uPred[i, :], vPred[i, :], pPred[i, :], X_pred[i, :], Y_pred[i, :], uRange, vRange, pRange)

            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            frames.append(Image.open(buf))
            
            plt.close()

        frames[0].save(gif_name, save_all=True, append_images=frames[1:], duration=500, loop=0)
        print(f"GIF saved as {gif_name}")


    def writeSolution(self, filename, x_data, y_data, variable_data, variables):
        if not all(len(x_data) == len(y_data) == len(var) for var in variable_data):
            raise ValueError("x_data, y_data, and all variables must have the same length.")
        
        with open(filename, 'w') as file:
            file.write(f"x coord, y coord, {', '.join(variables)}\n")
            
            for i in range(len(x_data)):
                row = (
                    [f"{x_data[i]:.4e}", f"{y_data[i]:.4e}"] +
                    [f"{var[i]:.4e}" for var in variable_data]
                )
                file.write(", ".join(row) + "\n")

        return


