from pinns import PINN
from problem.Burgers import BurgersEquation
from problem.Heat import HeatEquation2D
from problem.Wave import WaveEquation
from problem.NavierStokes import *
from problem.UnsteadyNavierStokes import *

def main():
    eq = 'UnsteadyFlowOverAirfoil'
    
    equations = {
        'Wave': (WaveEquation, (-1, 1), (0, 1), 'uniform'),
        'Heat': (HeatEquation2D, (-1, 1), (-1, 1), 'random'),
        'Burgers': (BurgersEquation, (-1, 1), (0, 1), 'random'),
        'LidDrivenCavity': (LidDrivenCavity, (-1, 1), (-1, 1), 'random'),
        'FlatPlate': (FlatPlate, (-3, 5), (-3, 3), 'random', 0.0),
        'FlowOverAirfoil': (FlowOverAirfoil, (-3, 5), (-3, 3), 'random', 0.0),
        'UnsteadyFlowOverAirfoil': (UnsteadyFlowOverAirfoil, (-3, 5), (-3, 3), (0, 10), 'random', 5.0, 1e2)
    }

    if eq not in equations:
        raise ValueError(f"Unsupported equation type: {eq}")

    if eq in ['UnsteadyFlowOverAirfoil']:
        equation_params = equations[eq]
        equation_class = equation_params[0]
        ranges = equation_params[1:4]
        sampling_method = equation_params[4]
    else:
        equation_params = equations[eq]
        equation_class = equation_params[0]
        ranges = equation_params[1:3]
        sampling_method = equation_params[3]
    
    if eq in ['FlatPlate', 'FlowOverAirfoil']:
        AoA = equation_params[4]
        equation = equation_class(AoA=AoA)
    elif eq in ['UnsteadyFlowOverAirfoil']:
        AoA = equation_params[5]
        reynoldsNumber = equation_params[6]
        equation = equation_class(AoA=AoA, Re = reynoldsNumber)
    else:
        equation = equation_class()

    
    if eq in ['FlatPlate', 'FlowOverAirfoil', 'LidDrivenCavity']:
        pinn = PINN(output_shape = 3, eq = eq)
    elif eq in ['UnsteadyFlowOverAirfoil']:
        pinn = PINN(input_shape = 3, output_shape = 3, eq = f'{eq}_AoA{AoA}_Re{reynoldsNumber}', layers = [20,40,40,40,20])
    else:
        pinn = PINN(eq = eq)

    data = equation.generate_data(*ranges, N0=1000, Nf=6000, sampling_method=sampling_method)
    pinn.train(equation.loss_function, data, print_interval=100, epochs=100000)

    return

if __name__ == "__main__":
    main()
