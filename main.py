from pinns import PINN
from problem.Burgers import BurgersEquation
from problem.Heat import HeatEquation2D
from plots import *

def main():
    # Create the Burgers' equation problem
    burgers_eq = BurgersEquation()

    # Define the domain
    x_range = (-8, 8)
    t_range = (0, 8)

    # Generate data
    data  = burgers_eq.generate_data(x_range, t_range, N0=100, Nf=10000, sampling_method='uniform')

    # Create the PINN model
    pinn = PINN()

    # Train the model
    pinn.train(burgers_eq.loss_function, data, epochs=10000, print_interval=1000)

    Nx = 5000
    Nt = 25000

    # Prediction grid
    x_pred = np.linspace(x_range[0], x_range[1], 100)[:, None].astype(np.float32)
    t_pred = np.linspace(t_range[0], t_range[1], 100)[:, None].astype(np.float32)
    X_pred, T_pred = np.meshgrid(x_pred, t_pred)

    uPred = pinn.model.predict(np.hstack((X_pred.flatten()[:, None], T_pred.flatten()[:, None]))) 
    
    x_num = np.linspace(x_range[0], x_range[1], Nx)[:, None].astype(np.float32)
    t_num = np.linspace(t_range[0], t_range[1], Nt + 1)[:, None].astype(np.float32)
    X_num, T_num = np.meshgrid(x_num, t_num)

    uNumeric = burgers_eq.numericalSolution(x_range, t_range, Nx, Nt) #Numerical solution to the Burguers equation

    plot = Plot(uPred, X_pred, T_pred, uNumeric, X_num, T_num)
    plot.contour_plot()

if __name__ == "__main__":
    main()
