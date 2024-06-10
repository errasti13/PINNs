from pinns import PINN
from Burgers import BurgersEquation
from Heat import HeatEquation2D
from plots import *

def main():
    # Create the Burgers' equation problem
    equation = HeatEquation2D()

    # Define the domain
    x_range = (0, 1)
    y_range = (0, 1)

    # Generate data
    x_f, y_f, xBc_left, yBc_left, uBc_left, xBc_right, yBc_right, uBc_right, xBc_bottom, yBc_bottom, uBc_bottom, xBc_top, yBc_top, uBc_top = equation.generate_data(x_range, y_range, N0=100, Nf=10000, sampling_method='uniform')

    # Create the PINN model
    pinn = PINN()

    # Train the model
    data = (x_f, y_f, xBc_left, yBc_left, uBc_left, xBc_right, yBc_right, uBc_right, xBc_bottom, yBc_bottom, uBc_bottom, xBc_top, yBc_top, uBc_top)
    pinn.train(equation.loss_function, data, epochs=1000, print_interval=100)

    Nx = 100
    Ny = 100

    # Prediction grid
    x_pred = np.linspace(x_range[0], x_range[1], 100)[:, None].astype(np.float32)
    y_pred = np.linspace(y_range[0], y_range[1], 100)[:, None].astype(np.float32)
    X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

    uPred = pinn.model.predict(np.hstack((X_pred.flatten()[:, None], Y_pred.flatten()[:, None]))) 
    
    x_num = np.linspace(x_range[0], x_range[1], Nx)[:, None].astype(np.float32)
    t_num = np.linspace(y_range[0], y_range[1], Ny)[:, None].astype(np.float32)
    X_num, T_num = np.meshgrid(x_num, t_num)

    uNumeric = equation.numericalSolution(x_range, y_range, Nx, Ny) #Numerical solution to the Burguers equation

    plot = Plot(uPred, X_pred, Y_pred, uNumeric, X_num, T_num)
    plot.contour_plot()

if __name__ == "__main__":
    main()
