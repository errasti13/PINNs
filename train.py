from pinns import PINN
from problem.Burgers import BurgersEquation
from problem.Heat import HeatEquation2D
from problem.Wave import WaveEquation
from problem.NavierStokes import *
from plots import *

def main():
    eq = 'Heat'
    
    pinn = PINN(output_shape=3, eq=eq)

    # Mapping equation types to their respective classes and parameters
    equations = {
        'Wave': (WaveEquation, (-1, 1), (0, 1), 'uniform'),
        'Heat': (HeatEquation2D, (-1, 1), (-1, 1), 'random'),
        'Burgers': (BurgersEquation, (-1, 1), (0, 1), 'random'),
        'LidDrivenCavity': (LidDrivenCavity, (-1, 1), (-1, 1), 'random'),
        'FlatPlate': (FlatPlate, (-3, 5), (-3, 3), 'random'),
        'FlowOverAirfoil': (FlowOverAirfoil, (-3, 5), (-3, 3), 'random')
    }

    if eq not in equations:
        raise ValueError(f"Unsupported equation type: {eq}")

    if eq == 'Heat' or eq == 'LidDrivenCavity' or eq == 'ChannelFlow' or eq == 'FlatPlate' or eq == 'FlowOverAirfoil':
        equation_class, x_range, y_range, sampling_method = equations[eq]
        if eq == 'FlatPlate' or eq == 'FlowOverAirfoil':
            AoA = 0.0
            equation = equation_class(AoA=AoA)
        else:
            equation = equation_class()

        data = equation.generate_data(x_range, y_range, N0=4000, Nf=4000, sampling_method=sampling_method)
    else:
        equation_class, x_range, t_range, sampling_method = equations[eq]
        equation = equation_class()
        data = equation.generate_data(x_range, t_range, N0=1000, Nf=100, sampling_method=sampling_method)

    pinn.train(equation.loss_function, data, print_interval=100, epochs=100000)

    if eq == 'Heat':
        uPred, X_pred, Y_pred, uNumeric, X_num, Y_num = equation.predict(pinn, x_range, y_range)
    elif eq == 'ChannelFlow' or eq == 'LidDrivenCavity' or eq == 'FlatPlate' or eq == 'FlowOverAirfoil':
        uPred, vPred, pPred, X_pred, Y_pred = equation.predict(pinn, x_range, y_range, Nx = 1024, Ny = 1024)
        equation.plot(X_pred, Y_pred, uPred, vPred, pPred)
    else:
        uPred, X_pred, T_pred, uNumeric, X_num, T_num = equation.predict(pinn, x_range, t_range)    

if __name__ == "__main__":
    main()
