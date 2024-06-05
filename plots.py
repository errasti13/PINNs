import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plot:
    def __init__(self, u_pred, X_pred, T_pred, x0, u0):
        self.u_pred = u_pred
        self.X_pred = X_pred
        self.T_pred = T_pred
        self.x0 = x0
        self.u0 = u0
        self.U_pred = self.u_pred.reshape(self.X_pred.shape)
        
    def contour_plot(self):
        plt.figure(figsize=(10, 6))
        plt.contourf(self.T_pred, self.X_pred, self.U_pred, levels=100, cmap='jet')
        plt.colorbar()
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('Predicted u(x,t)')
        plt.show()
    
    def surface_plot(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.T_pred, self.X_pred, self.U_pred, cmap='jet')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('u(x,t)')
        ax.set_title('3D Surface plot of u(x,t)')
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
    
    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
