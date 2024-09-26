from pinns import PINN
from problem.Burgers import BurgersEquation
from problem.Heat import HeatEquation2D
from problem.Wave import WaveEquation
from plots import *

def main():
    eq = 'Burgers'
    
    pinn = PINN()

    # Mapping equation types to their respective classes and parameters
    equations = {
        'Wave': (WaveEquation, (-1, 1), (0, 1), 'uniform'),
        'Heat': (HeatEquation2D, (-1, 1), (-1, 1), 'random'),
        'Burgers': (BurgersEquation, (-1, 1), (0, 1), 'random'),
    }

    if eq not in equations:
        raise ValueError(f"Unsupported equation type: {eq}")

    equation_class, x_range, t_range, sampling_method = equations[eq]
    equation = equation_class()

    if eq == 'Heat':
        data = equation.generate_data(x_range, y_range=(-1, 1), N0=100, Nf=10000, sampling_method=sampling_method)
    else:
        data = equation.generate_data(x_range, t_range, N0=100, Nf=10000, sampling_method=sampling_method)

    pinn.train(equation.loss_function, data, epochs=10000 if eq == 'Burgers' else 3000, print_interval=1000)

    if eq == 'Heat':
        uPred, X_pred, Y_pred, uNumeric, X_num, Y_num = equation.predict(pinn, x_range, (-1, 1))
        plot = Plot(uPred, X_pred, Y_pred, uNumeric, X_num, Y_num)
    else:
        uPred, X_pred, T_pred, uNumeric, X_num, T_num = equation.predict(pinn, x_range, t_range)
        plot = Plot(uPred, X_pred, T_pred, uNumeric, X_num, T_num)

    plot.contour_plot()

if __name__ == "__main__":
    main()
