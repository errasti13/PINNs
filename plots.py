import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self, u_pred, X_pred, T_pred, u_num, X_num, T_num):

        self.X_pred = X_pred
        self.T_pred = T_pred
        self.U_pred = u_pred.reshape(self.X_pred.shape)

        self.X_num = X_num
        self.T_num = T_num
        self.U_num = u_num.reshape(self.X_num.shape)

    def contour_plot(self):
        plt.figure(figsize=(10, 6))

        # Predicted solution
        plt.subplot(1, 2, 1)
        plt.contourf(self.T_pred, self.X_pred, self.U_pred, levels=100, cmap='jet')
        plt.colorbar()
        plt.title('PINN Solution')

        # Numerical solution
        
        plt.subplot(1, 2, 2)
        plt.contourf(self.T_num, self.X_num, self.U_num, levels=100, cmap='jet')
        plt.colorbar()
        plt.title('Numerical Solution')

        plt.tight_layout()
        plt.show()

    def plotNumericalSolution(self):

        # Create meshgrid for plotting
        x = np.linspace(self.X_pred.min(), self.X_pred.min(), self.Nx)
        t = np.linspace(self.T_pred.min(), self.T_pred.max()[1], self.Nt+1)
        X, T = np.meshgrid(x, t)

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.contourf(T, X, self.u_numeric, levels=100, cmap='jet')
        plt.colorbar()
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('Contour Plot of Numerical Solution')
        plt.show()
        plt.show()
    
    def predicted_data_plot(self):
        plt.figure(figsize=(6, 4))
        
        time_range = self.T_pred.max() - self.T_pred.min()
        time_points = [0, 0.25, 0.5, 0.75, 1]
        
        for value in time_points:
            idx = self.find_nearest(self.T_pred.flatten(), time_range * value)
            plt.plot(self.X_pred.flatten()[idx], self.u_pred.flatten()[idx], label=f'Predicted at t={time_range * value}')
        
        plt.xlabel('X')
        plt.ylabel('u')
        plt.title('Predicted Data')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def compareU(self, time):
        idxPred = self.find_nearest(self.T_pred.flatten(), time)
        idxNum = self.find_nearest(self.T_num.flatten(), time)

        plt.figure(figsize=(10, 6))
        plt.plot(self.X_num.flatten()[idxNum], self.U_num.flatten()[idxNum], 'b-', label='Numerical Solution')
        plt.plot(self.X_pred.flatten()[idxPred], self.U_pred.flatten()[idxPred], 'r--', label='PINN Solution')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title(f'Comparison of u(x) at t = {time}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        diff = np.abs(array - value)
        min_diff = np.min(diff)
        idx = np.where(diff == min_diff)[0]
        return idx

class PlotNSSolution:
    def __init__(self, u_pred, v_pred, p_pred, X_pred, Y_pred):

        self.X_pred = X_pred
        self.Y_pred = Y_pred

        self.U_pred = u_pred.reshape(self.X_pred.shape)
        self.V_pred = v_pred.reshape(self.X_pred.shape)
        self.P_pred = p_pred.reshape(self.X_pred.shape)

    def contour_plot(self):
        plt.figure(figsize=(16, 8))

        # Coordinates for the plate (x from 0 to 1, y fixed at 0)
        xPlate = np.linspace(0.0, 1.0, 1000)[:, None].astype(np.float32)
        yPlate = np.full((1000, 1), 0.0, dtype=np.float32)

        plt.subplot(3, 1, 1)
        plt.contourf(self.X_pred, self.Y_pred, self.U_pred, levels=200, cmap='jet')
        plt.plot(xPlate, yPlate, color='black', linewidth=3)  # Thick black line
        plt.colorbar()
        plt.title('U')

        plt.subplot(3, 1, 2)
        plt.contourf(self.X_pred, self.Y_pred, self.V_pred, levels=200, cmap='jet')
        plt.plot(xPlate, yPlate, color='black', linewidth=3)  # Thick black line
        plt.colorbar()
        plt.title('V')

        plt.subplot(3, 1, 3)
        plt.contourf(self.X_pred, self.Y_pred, self.P_pred, levels=200, cmap='jet')
        plt.plot(xPlate, yPlate, color='black', linewidth=3)  # Thick black line
        plt.colorbar()
        plt.title('P')

        plt.tight_layout()
        plt.show()
