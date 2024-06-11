import numpy as np

class ConditionsLibrary:
    def __init__(self, conditions):
        self.conditions = conditions
    
    def getCondition(self, N0, x, y, condition='zeros', **kwargs):
        if condition in self.conditions:
            condition_func = self.conditions[condition]
            xCond, uCond = condition_func(N0, x, y, **kwargs)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        return xCond, uCond

class BoundaryConditionLibrary(ConditionsLibrary):
    def __init__(self):
        conditions = {
            'zeros': zeros1D,
            'ones': ones1D,
            'senoidal': senoidal1D,
            'linear': linear1D,
            'quadratic': quadratic1D,
            'exponential': exponential1D
        }
        super().__init__(conditions)

class InitialConditionsLibrary(ConditionsLibrary):
    def __init__(self):
        conditions = {
            'zeros': zeros1D,
            'ones': ones1D,
            'senoidal': senoidal1D,
            'linear': linear1D,
            'quadratic': quadratic1D,
            'exponential': exponential1D
        }
        super().__init__(conditions)

def zeros1D(N0, x, y):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = np.zeros((N0, 1), dtype=np.float32)
    return xBc, uBc

def ones1D(N0, x, y):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = np.ones((N0, 1), dtype=np.float32)
    return xBc, uBc
    
def senoidal1D(N0, x, y):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = -np.sin(np.pi * y).astype(np.float32)  
    return xBc, uBc

def linear1D(N0, x, y, slope=1.0, intercept=0.0):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = (slope * y + intercept).astype(np.float32)
    return xBc, uBc

def quadratic1D(N0, x, y, a=1.0, b=0.0, c=0.0):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = (a * y**2 + b * y + c).astype(np.float32)
    return xBc, uBc

def exponential1D(N0, x, y, a=1.0, b=1.0):
    xBc = np.full((N0, 1), x, dtype=np.float32)
    uBc = (a * np.exp(b * y)).astype(np.float32)
    return xBc, uBc
