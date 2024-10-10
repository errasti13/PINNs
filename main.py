from pinns import PINN
from problem.Burgers import BurgersEquation
from problem.Heat import HeatEquation2D
from problem.Wave import WaveEquation
from problem.NavierStokes import *
from plots import *

def main():
    eq = 'FlatPlate'
    
    pinn = PINN(output_shape=3)

    # Mapping equation types to their respective classes and parameters
    equations = {
        'Wave': (WaveEquation, (-1, 1), (0, 1), 'uniform'),
        'Heat': (HeatEquation2D, (-1, 1), (-1, 1), 'random'),
        'Burgers': (BurgersEquation, (-1, 1), (0, 1), 'random'),
        'LidDrivenCavity': (LidDrivenCavity, (-1, 1), (-1, 1), 'random'),
        'ChannelFlow': (ChannelFlow, (0, 10), (0, 1), 'random'),
        'FlatPlate': (FlatPlate, (-5, 5), (-5, 5), 'random')
    }

    if eq not in equations:
        raise ValueError(f"Unsupported equation type: {eq}")

    if eq == 'Heat' or eq == 'LidDrivenCavity' or eq == 'ChannelFlow' or eq == 'FlatPlate':
        equation_class, x_range, y_range, sampling_method = equations[eq]
        if eq == 'FlatPlate':
            AoA = 0.0
            equation = equation_class(AoA=AoA)
        else:
            equation = equation_class()

        data = equation.generate_data(x_range, y_range, N0=5000, Nf=5000, sampling_method=sampling_method)
    else:
        equation_class, x_range, t_range, sampling_method = equations[eq]
        equation = equation_class()
        data = equation.generate_data(x_range, t_range, N0=100, Nf=10000, sampling_method=sampling_method)

    pinn.train(equation.loss_function, data, print_interval=100, epochs=1000)
    pinn.model.save(f'trainedModels/{eq}.tf')

    if eq == 'Heat':
        uPred, X_pred, Y_pred, uNumeric, X_num, Y_num = equation.predict(pinn, x_range, y_range)
        plot = Plot(uPred, X_pred, Y_pred, uNumeric, X_num, Y_num)
    elif eq == 'ChannelFlow' or eq == 'LidDrivenCavity' or eq == 'FlatPlate':
        uPred, vPred, pPred, X_pred, Y_pred = equation.predict(pinn, x_range, y_range, Nx = 512, Ny = 512)
        plot = PlotNSSolution(uPred, vPred, pPred, X_pred, Y_pred)
    else:
        uPred, X_pred, T_pred, uNumeric, X_num, T_num = equation.predict(pinn, x_range, t_range)
        plot = Plot(uPred, X_pred, T_pred, uNumeric, X_num, T_num)

    plot.contour_plot()

if __name__ == "__main__":
    main()
