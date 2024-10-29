import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class UnsteadyNavierStokes2D:
    
    def __init__(self, nu=0.01):
        self.nu = nu

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
    
    def __init__(self, nu=0.01, c=1, AoA=0.0, uInlet=1.0, airfoil_coords=None):
        super().__init__(nu)
        self.problemTag = "FlowOverAirfoil"
        self.c = c  
        self.AoA = AoA * np.pi / 180 
        self.uInlet = uInlet
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

    def getBoundaryCondition(self, N0, Nf, x_min, x_max, y_min, y_max, t_min, t_max, sampling_method='uniform', xLE = 0.0, yLE = 0.0):
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
            boundaries['left']['x'] = np.full((Nf, 1), x_min, dtype=np.float32)
            boundaries['left']['y'] = (np.random.rand(Nf, 1) * (y_max - y_min) + y_min).astype(np.float32)
            boundaries['left']['t'] = (np.random.rand(N0, 1) * (t_max - t_min) + t_min).astype(np.float32)

            boundaries['right']['x'] = np.full((Nf, 1), x_max, dtype=np.float32)
            boundaries['right']['y'] = (np.random.rand(Nf, 1) * (y_max - y_min) + y_min).astype(np.float32)
            boundaries['right']['t'] = (np.random.rand(N0, 1) * (t_max - t_min) + t_min).astype(np.float32)

            boundaries['bottom']['y'] = np.full((Nf, 1), y_min, dtype=np.float32)
            boundaries['bottom']['x'] = (np.random.rand(Nf, 1) * (x_max - x_min) + x_min).astype(np.float32)
            boundaries['bottom']['t'] = (np.random.rand(N0, 1) * (t_max - t_min) + t_min).astype(np.float32)

            boundaries['top']['y'] = np.full((Nf, 1), y_max, dtype=np.float32)
            boundaries['top']['x'] = (np.random.rand(Nf, 1) * (x_max - x_min) + x_min).astype(np.float32)
            boundaries['top']['t'] = (np.random.rand(N0, 1) * (t_max - t_min) + t_min).astype(np.float32)

            boundaries['initial']['y'] = (np.random.rand(Nf, 1) * (y_max - y_min) + y_min).astype(np.float32)
            boundaries['initial']['x'] = (np.random.rand(Nf, 1) * (x_max - x_min) + x_min).astype(np.float32)
            boundaries['initial']['t'] = np.full((Nf, 1), t_min, dtype=np.float32).astype(np.float32)

            boundaries['airfoil']['t'] = (np.random.rand(N0, 1) * (t_max - t_min) + t_min).astype(np.float32)

        elif sampling_method == 'uniform':
            yBc = np.linspace(y_min, y_max, Nf)[:, None].astype(np.float32)
            xBc = np.linspace(x_min, x_max, Nf)[:, None].astype(np.float32)
            tBc = np.linspace(t_min, t_max, N0)[:, None].astype(np.float32)

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

        boundaries['initial']['u'] = tf.zeros_like(boundaries['initial']['t'], dtype=np.float32)
        boundaries['initial']['v'] = tf.zeros_like(boundaries['initial']['t'], dtype=np.float32)
        boundaries['initial']['p'] = tf.zeros_like(boundaries['initial']['t'], dtype=np.float32)

        return boundaries
    
    def generate_data(self, x_range, y_range, t_range, N0=100, Nf=10000, sampling_method='random'):
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]
        t_min, t_max = t_range[0], t_range[1]

        boundaries = self.getBoundaryCondition(N0, Nf, x_min, x_max, y_min, y_max, t_min, t_max, sampling_method)

        x_f, y_f, t_f = [], [], []
        while len(x_f) < Nf:
            x_candidate = (np.random.rand(1) * (x_max - x_min) + x_min).astype(np.float32)
            y_candidate = (np.random.rand(1) * (y_max - y_min) + y_min).astype(np.float32)
            t_candidate = (np.random.rand(1) * (y_max - y_min) + y_min).astype(np.float32)
            
            if not self.is_point_inside_airfoil(x_candidate, y_candidate):
                x_f.append(x_candidate)
                y_f.append(y_candidate)
                t_f.append(t_candidate)

        x_f = np.array(x_f, dtype=np.float32).reshape(-1, 1)
        y_f = np.array(y_f, dtype=np.float32).reshape(-1, 1)
        t_f = np.array(t_f, dtype=np.float32).reshape(-1, 1)

        return x_f, y_f, t_f, boundaries
    
    def predict(self, pinn, x_range, y_range, Nx=256, Ny=256):
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]
        Nt = Nx * Ny  

        x_pred, y_pred = [], []
        while len(x_pred) < Nt:
            x_candidate = (np.random.rand(1) * (x_max - x_min) + x_min).astype(np.float32)
            y_candidate = (np.random.rand(1) * (y_max - y_min) + y_min).astype(np.float32)

            if not self.is_point_inside_airfoil(x_candidate, y_candidate):
                x_pred.append(x_candidate)
                y_pred.append(y_candidate)

        x_pred = np.array(x_pred, dtype=np.float32).reshape(-1, 1)
        y_pred = np.array(y_pred, dtype=np.float32).reshape(-1, 1)

        predictions = pinn.predict(np.hstack((x_pred.flatten()[:, None], y_pred.flatten()[:, None])))

        uPred, vPred, pPred = predictions[:, 0], predictions[:, 1], predictions[:, 2]

        return uPred, vPred, pPred, x_pred, y_pred

    def plot(self, X_pred, Y_pred, uPred, vPred, pPred):
        plt.figure(figsize=(16, 8))

        # Plot uPred
        plt.subplot(3, 1, 1)
        plt.plot(self.xAirfoil, self.yAirfoil, color='black')
        plt.plot(self.xAirfoil, -self.yAirfoil, color='black')
        plt.scatter(X_pred, Y_pred, c=uPred, cmap='jet', s=2)
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
        plt.scatter(X_pred, Y_pred, c=vPred, cmap='jet', s=2)
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
        plt.scatter(X_pred, Y_pred, c=pPred, cmap='jet', s=2)
        plt.xlim(X_pred.min(), X_pred.max())
        plt.ylim(Y_pred.min(), Y_pred.max())
        plt.colorbar()
        plt.title('Predicted Pressure')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.tight_layout()
        plt.show()

