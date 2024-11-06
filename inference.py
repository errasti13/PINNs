from pinns import PINN
from problem.Burgers import BurgersEquation
from problem.Heat import HeatEquation2D
from problem.Wave import WaveEquation
from problem.NavierStokes import *
from problem.UnsteadyNavierStokes import *

def main():
    eq = 'UnsteadyFlowOverAirfoil'
    
    pinn = PINN(eq=eq)
    equations = {
        'Wave': (WaveEquation, (-1, 1), (0, 1), 'uniform'),
        'Heat': (HeatEquation2D, (-1, 1), (-1, 1), 'random'),
        'Burgers': (BurgersEquation, (-1, 1), (0, 1), 'random'),
        'LidDrivenCavity': (LidDrivenCavity, (-1, 1), (-1, 1), 'random'),
        'FlatPlate': (FlatPlate, (-3, 5), (-3, 3), 'random', 0.0),
        'FlowOverAirfoil': (FlowOverAirfoil, (-3, 5), (-3, 3), 'random', 0.0),
        'UnsteadyFlowOverAirfoil': (UnsteadyFlowOverAirfoil, (-3, 5), (-3, 3), (0, 1), 'random', 5.0, 1000)
    }

    if eq not in equations:
        raise ValueError(f"Unsupported equation type: {eq}")

    equation_params = equations[eq]
    equation_class = equation_params[0]
    ranges = equation_params[1:3]
    sampling_method = equation_params[3]
    
    if eq in ['FlatPlate', 'FlowOverAirfoil']:
        equation_params = equations[eq]
        equation_class = equation_params[0]
        ranges = equation_params[1:3]
        sampling_method = equation_params[3]

        AoA = equation_params[4]
        equation = equation_class(AoA=AoA)
    elif eq in ['UnsteadyFlowOverAirfoil']:
        equation_params = equations[eq]
        equation_class = equation_params[0]
        ranges = equation_params[1:4]
        sampling_method = equation_params[4]
        
        AoA = equation_params[5]
        reynoldsNumber = equation_params[6]
        equation = equation_class(AoA=AoA, Re = reynoldsNumber)
    else:
        equation = equation_class()

    if eq in ['UnsteadyFlowOverAirfoil']:
        pinn.model = tf.keras.models.load_model(f'trainedModels/{eq}' + f'_AoA{AoA}' + f'_Re{reynoldsNumber}.tf')
    else:
        pinn.model = tf.keras.models.load_model(f'trainedModels/{eq}.tf')

    if eq == 'Burgers':
        equation.predict(pinn, ranges[0], ranges[1], Nx = 100, Nt = 100)
        equation.computeNumerical(ranges[0], ranges[1], Nx = 2000, Nt = 10000)
        equation.plot()

    if eq == 'Heat':
        equation.predict(pinn, ranges[0], ranges[1], Nx = 1000, Ny = 1000)
        equation.computeNumerical(ranges[0], ranges[1], Nx = 200, Ny = 200)
        equation.plot()

    if eq == 'UnsteadyFlowOverAirfoil':
        uPred_all, vPred_all, pPred_all, x_pred_all, y_pred_all, t_pred_all = equation.predict(pinn, *ranges, Nx = 512, Ny = 512, Nt = 100)
        equation.create_gif_from_plots(uPred_all, vPred_all, pPred_all, x_pred_all, y_pred_all, t_pred_all)
        

if __name__ == "__main__":
    main()