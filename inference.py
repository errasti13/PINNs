from pinns import PINN
from problem.Burgers import BurgersEquation
from problem.Heat import HeatEquation2D
from problem.Wave import WaveEquation
from problem.NavierStokes import *

def main():
    eq = 'Heat'
    
    pinn = PINN(eq=eq)
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
    
    if eq in ['FlatPlate', 'FlowOverAirfoil']:
        AoA = equation_params[4]
        equation = equation_class(AoA=AoA)
    else:
        equation = equation_class()

    pinn.model = tf.keras.models.load_model(f'trainedModels/{eq}.tf')

    
    if eq == 'Burgers':
        equation.predict(pinn, ranges[0], ranges[1], Nx = 100, Nt = 100)
        equation.computeNumerical(ranges[0], ranges[1], Nx = 2000, Nt = 10000)
        equation.plot()

    if eq == 'Heat':
        equation.predict(pinn, ranges[0], ranges[1], Nx = 1000, Ny = 1000)
        equation.computeNumerical(ranges[0], ranges[1], Nx = 200, Ny = 200)
        equation.plot()

if __name__ == "__main__":
    main()