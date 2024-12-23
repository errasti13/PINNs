import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class SteadyNavierStokes2D:
    
    def __init__(self, nu=0.01):
        self.nu = nu

    def getBoundaryCondition(self, N0, x_min, x_max, y_min, y_max, sampling_method='uniform'):
        raise NotImplementedError("Boundary condition method must be implemented in a subclass.")
    
    def generate_data(self, x_range, y_range, N0=100, Nf=10000, sampling_method='random'):
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]

        boundaries = self.getBoundaryCondition(N0, x_min, x_max, y_min, y_max, sampling_method)

        if sampling_method == 'random':
            x_f = (np.random.rand(Nf, 1) * (x_max - x_min) + x_min).astype(np.float32)
            y_f = (np.random.rand(Nf, 1) * (y_max - y_min) + y_min).astype(np.float32)
        elif sampling_method == 'uniform':
            x_f = np.linspace(x_min, x_max, Nf)[:, None].astype(np.float32)
            y_f = np.linspace(y_min, y_max, Nf)[:, None].astype(np.float32)

        return x_f, y_f, boundaries
    
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
        x_f, y_f, boundaries = data

        total_loss = 0

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x_f, y_f])

            uvp_pred = model(tf.concat([x_f, y_f], axis=1))
            u_pred = uvp_pred[:, 0]
            v_pred = uvp_pred[:, 1]
            p_pred = uvp_pred[:, 2]

            u_x = tape.gradient(u_pred, x_f)
            u_y = tape.gradient(u_pred, y_f)
            v_x = tape.gradient(v_pred, x_f)
            v_y = tape.gradient(v_pred, y_f)
            p_x = tape.gradient(p_pred, x_f)
            p_y = tape.gradient(p_pred, y_f)

            u_xx = tape.gradient(u_x, x_f)
            u_yy = tape.gradient(u_y, y_f)
            v_xx = tape.gradient(v_x, x_f)
            v_yy = tape.gradient(v_y, y_f)

        continuity = u_x + v_y

        momentum_u = u_pred * u_x + v_pred * u_y + p_x - self.nu * (u_xx + u_yy)

        momentum_v = u_pred * v_x + v_pred * v_y + p_y - self.nu * (v_xx + v_yy)

        f_loss_u = tf.reduce_mean(tf.square(momentum_u))
        f_loss_v = tf.reduce_mean(tf.square(momentum_v))
        continuity_loss = tf.reduce_mean(tf.square(continuity))

        total_loss += f_loss_u + f_loss_v + continuity_loss

        for boundary_key, boundary_data in boundaries.items():
            xBc = boundary_data['x']
            yBc = boundary_data['y']
            uBc = boundary_data['u']
            vBc = boundary_data['v']
            pBc = boundary_data['p']

            uBc_tensor, vBc_tensor, pBc_tensor = self.imposeBoundaryCondition(uBc, vBc, pBc)
            uBc_loss, vBc_loss, pBc_loss = self.computeBoundaryLoss(model, xBc, yBc, uBc_tensor, vBc_tensor, pBc_tensor)

            total_loss += uBc_loss + vBc_loss + pBc_loss

        return total_loss

    def predict(self, pinn, x_range, y_range, Nx=256, Ny=256):
        x_pred = np.linspace(x_range[0], x_range[1], Nx)[:, None].astype(np.float32)
        y_pred = np.linspace(y_range[0], y_range[1], Ny)[:, None].astype(np.float32)
        X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

        predictions = pinn.predict(np.hstack((X_pred.flatten()[:, None], Y_pred.flatten()[:, None])))

        uPred, vPred, pPred = predictions[:, 0], predictions[:, 1], predictions[:, 2]

        return uPred, vPred, pPred, X_pred, Y_pred

class LidDrivenCavity(SteadyNavierStokes2D):
    
    def __init__(self, nu=0.01):
        super().__init__(nu)
        self.problemTag = "LidDrivenCavity"

        return
    
    def getBoundaryCondition(self, N0, x_min, x_max, y_min, y_max, sampling_method='uniform'):
        boundaries = {
            'left': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'right': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'bottom': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'top': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None}
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

        for key in boundaries:
            boundaries[key]['u'] = np.zeros_like(boundaries[key]['x'], dtype=np.float32) 
            boundaries[key]['v'] = np.zeros_like(boundaries[key]['y'], dtype=np.float32)

        boundaries['top']['u'] = np.ones_like(boundaries['top']['x'], dtype=np.float32)

        return boundaries

class FlatPlate(SteadyNavierStokes2D):
    
    def __init__(self, nu=0.01, c = 1, AoA = 0.0, uInlet = 1.0):
        super().__init__(nu)
        self.problemTag = "ChannelFlow"
        self.xLE = 0.0
        self.yLE = 0.0
        self.c = c
        self.AoA = AoA * np.pi / 180
        self.uInlet = uInlet

        self.generate_airfoil_coords()

        return
    
    def generate_airfoil_coords(self, N=100):
        x = np.random.rand(N, 1) * self.c + self.xLE

        y =  np.full((N, 1), self.yLE, dtype=np.float32)
        
        self.xAirfoil = x.reshape(-1, 1)
        self.yAirfoil = y.reshape(-1, 1)

        return
    
    def getBoundaryCondition(self, N0, x_min, x_max, y_min, y_max, sampling_method='uniform', xLE = 0.0, yLE = 0.0):
        boundaries = {
            'left': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'right': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'bottom': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'top': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'plate': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None}
        }

        boundaries['plate']['y'] = self.yAirfoil
        boundaries['plate']['x'] = self.xAirfoil

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
        
        boundaries['left']['u'] = self.uInlet * np.cos(self.AoA)*tf.ones_like(boundaries['left']['x'], dtype=np.float32)
        boundaries['left']['v'] = self.uInlet * np.sin(self.AoA)*tf.ones_like(boundaries['left']['y'], dtype=np.float32)
        boundaries['left']['p'] = tf.zeros_like(boundaries['right']['x'], dtype=np.float32)

        boundaries['top']['u'] = self.uInlet * np.cos(self.AoA)*tf.ones_like(boundaries['top']['x'], dtype=np.float32)
        boundaries['top']['v'] = self.uInlet * np.sin(self.AoA)*tf.ones_like(boundaries['top']['y'], dtype=np.float32)

        boundaries['right']['u'] = None
        boundaries['right']['v'] = None

        boundaries['bottom']['u'] = self.uInlet * np.cos(self.AoA)*tf.ones_like(boundaries['bottom']['x'], dtype=np.float32)
        boundaries['bottom']['v'] = self.uInlet * np.sin(self.AoA)*tf.ones_like(boundaries['bottom']['y'], dtype=np.float32)

        boundaries['plate']['u'] = tf.zeros_like(boundaries['plate']['x'], dtype=np.float32)
        boundaries['plate']['v'] = tf.zeros_like(boundaries['plate']['y'], dtype=np.float32)

        return boundaries

class FlowOverAirfoil(SteadyNavierStokes2D):
    
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

    def getBoundaryCondition(self, N0, x_min, x_max, y_min, y_max, sampling_method='uniform', xLE = 0.0, yLE = 0.0):
        boundaries = {
            'left': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'right': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'bottom': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'top': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None},
            'airfoil': {'x': None, 'y': None, 'u': None, 'v': None, 'p': None}
        }
    
        boundaries['airfoil']['y'] = self.yAirfoil
        boundaries['airfoil']['x'] = self.xAirfoil

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

            boundaries['plate']['y'] = np.full((N0, 1), yLE, dtype=np.float32) 
            boundaries['plate']['x'] = np.linspace(xLE, xLE + self.c, N0)[:, None].astype(np.float32)
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

        return boundaries
    
    def generate_data(self, x_range, y_range, N0=100, Nf=10000, sampling_method='random'):
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]

        boundaries = self.getBoundaryCondition(N0, x_min, x_max, y_min, y_max, sampling_method)

        x_f, y_f = [], []
        while len(x_f) < Nf:
            x_candidate = (np.random.rand(1) * (x_max - x_min) + x_min).astype(np.float32)
            y_candidate = (np.random.rand(1) * (y_max - y_min) + y_min).astype(np.float32)
            
            if not self.is_point_inside_airfoil(x_candidate, y_candidate):
                x_f.append(x_candidate)
                y_f.append(y_candidate)

        x_f = np.array(x_f, dtype=np.float32).reshape(-1, 1)
        y_f = np.array(y_f, dtype=np.float32).reshape(-1, 1)

        return x_f, y_f, boundaries
    
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
