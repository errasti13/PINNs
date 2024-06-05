import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self, u_pred, X_pred, T_pred, x0, u0, u_num, X_num, T_num):

        self.x0 = x0
        self.u0 = u0

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
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('Predicted u(x,t)')

        # Numerical solution
        
        plt.subplot(1, 2, 2)
        plt.contourf(self.T_num, self.X_num, self.U_num, levels=100, cmap='jet')
        plt.colorbar()
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('Numerical u(x,t)')

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

    def initial_vs_predicted(self):
        idx = self.find_nearest(self.T_pred.flatten(), 0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.x0, self.u0, 'bo', label='Initial condition')
        plt.plot(self.X_pred.flatten()[idx], self.u_pred.flatten()[idx], 'r-', label='Predicted at t=0')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.title('Initial Condition vs Predicted Solution at t=0')
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


