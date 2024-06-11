import numpy as np

class BoundaryConditionLibrary:
    def __init__(self):
        self.boundary_conditions = {
            'zeros': zeros1D,
            'ones': ones1D,
            'senoidal': senoidal1D,
            'linear': linear1D,
            'quadratic': quadratic1D,
            'exponential': exponential1D
        }
    
    def getBoundaryCondition(self, N0, x, boundaryCondition='zeros'):

        if boundaryCondition in self.boundary_conditions:
            boundary_func = self.boundary_conditions[boundaryCondition]
            xBc, uBc = boundary_func(N0, x)
        else:
            raise ValueError(f"Unknown boundary condition: {boundaryCondition}")

        return xBc, uBc

def zeros1D(N0, x):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = np.zeros((N0, 1), dtype=np.float32)
    return xBc, uBc

def ones1D(N0, x):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = np.ones((N0, 1), dtype=np.float32)
    return xBc, uBc
    
def senoidal1D(N0, x):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = np.sin(np.pi * x).astype(np.float32)  
    return xBc, uBc

def linear1D(N0, x, slope=1.0, intercept=0.0):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = (slope * x + intercept).astype(np.float32)
    return xBc, uBc

def quadratic1D(N0, x, a=1.0, b=0.0, c=0.0):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = (a * x**2 + b * x + c).astype(np.float32)
    return xBc, uBc

def exponential1D(N0, x, a=1.0, b=1.0):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = (a * np.exp(b * x)).astype(np.float32)
    return xBc, uBc
