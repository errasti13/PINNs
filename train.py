from pinns import PINN
from problem.Burgers import BurgersEquation
from problem.Heat import HeatEquation2D
from problem.Wave import WaveEquation
from problem.NavierStokes import *

def main():
    eq = 'Heat'
    
    equations = {
        'Wave': (WaveEquation, (-1, 1), (0, 1), 'uniform'),
        'Heat': (HeatEquation2D, (-1, 1), (-1, 1), 'random'),
        'Burgers': (BurgersEquation, (-1, 1), (0, 1), 'random'),
        'LidDrivenCavity': (LidDrivenCavity, (-1, 1), (-1, 1), 'random'),
        'FlatPlate': (FlatPlate, (-3, 5), (-3, 3), 'random', 0.0),
        'FlowOverAirfoil': (FlowOverAirfoil, (-3, 5), (-3, 3), 'random', 0.0)
    }

    if eq not in equations:
        raise ValueError(f"Unsupported equation type: {eq}")

    equation_params = equations[eq]
    equation_class = equation_params[0]
    ranges = equation_params[1:3]
    sampling_method = equation_params[3]

    if eq == ['FlatPlate', 'FlowOverAirfoil', 'LidDrivenCavity']:
        pinn = PINN(output_shape = 3, eq = eq)
    else:
        pinn = PINN(eq = eq)
    
    if eq in ['FlatPlate', 'FlowOverAirfoil']:
        AoA = equation_params[4]
        equation = equation_class(AoA=AoA)
    else:
        equation = equation_class()

    data = equation.generate_data(*ranges, N0=4000, Nf=4000, sampling_method=sampling_method)
    pinn.train(equation.loss_function, data, print_interval=100, epochs=100000)

    return

if __name__ == "__main__":
    main()
