import numpy as np
from pinns import PINN
from problem import BurgersEquation
from plots import *

def main():
    # Create the Burgers' equation problem
    burgers_eq = BurgersEquation()

    # Define the domain
    x_range = (-1, 1)
    t_range = (0, 1)

    # Generate data
    x_f, t_f, u0, x0, t0, xBc0, tBc0, uBc0, xBc1, tBc1, uBc1  = burgers_eq.generate_data(x_range, t_range, N0=100, Nf=10000, sampling_method='uniform')

    # Create the PINN model
    pinn = PINN()

    # Train the model
    pinn.train(burgers_eq.loss_function, (x_f, t_f, u0, x0, t0, xBc0, tBc0, uBc0, xBc1, tBc1, uBc1), epochs=10000, print_interval=100)

    # Prediction grid
    x_pred = np.linspace(x_range[0], x_range[1], 100)[:, None].astype(np.float32)
    t_pred = np.linspace(t_range[0], t_range[1], 100)[:, None].astype(np.float32)
    X_pred, T_pred = np.meshgrid(x_pred, t_pred)

    # Visualize the solution
    visualize_solution(pinn, X_pred, T_pred, x0, u0)

if __name__ == "__main__":
    main()
