# Physics-Informed Neural Networks (PINNs) for Fluid Dynamics Simulation

## Overview

This project implements Physics-Informed Neural Networks (PINNs) to simulate and solve various fluid dynamics and partial differential equations (PDEs). The repository contains scripts for training neural network models and performing inference on different fluid flow scenarios.

## Features

The project supports simulation of the following fluid dynamics and mathematical models:
- Wave Equation
- Heat Equation (2D)
- Burgers' Equation
- Lid-Driven Cavity Flow
- Flat Plate Flow
- Flow Over Airfoil
- Unsteady Flow Over Airfoil

## Requirements

- Python
- TensorFlow
- NumPy
- Matplotlib
- Custom `pinns` module
- Custom problem-specific modules (`problem` directory)

## Project Structure

- `train.py`: Script for training Physics-Informed Neural Networks
- `inference.py`: Script for performing inference and generating predictions
- `problem/`: Directory containing equation-specific implementations
  - `Burgers.py`
  - `Heat.py`
  - `Wave.py`
  - `NavierStokes.py`
  - `UnsteadyNavierStokes.py`

## Usage

### Training a Model

To train a model for a specific equation, modify the `eq` variable in `train.py`:

```python
eq = 'UnsteadyFlowOverAirfoil'  # Change this to select different equations
```

Run the training script:

```bash
python train.py
```

### Performing Inference

Similarly, in `inference.py`, set the equation type:

```python
eq = 'UnsteadyFlowOverAirfoil'  # Change this to match your trained model
```

Run the inference script:

```bash
python inference.py
```

## Key Configuration Options

- Supported sampling methods: 'uniform', 'random'
- Configurable parameters include:
  - Angle of Attack (AoA)
  - Reynolds Number
  - Spatial and temporal ranges
  - Neural network layer configurations

## Output

- Trained models are saved in the `trainedModels/` directory
- For certain equations (e.g., Unsteady Flow Over Airfoil), the script generates:
  - CSV files with solution data
  - Animated GIF visualizations of flow dynamics

## Supported Equations

### Fluid Dynamics
- Lid-Driven Cavity Flow
- Flat Plate Flow
- Flow Over Airfoil
- Unsteady Flow Over Airfoil

### Mathematical PDEs
- Wave Equation
- Heat Equation (2D)
- Burgers' Equation

## Customization

You can extend the project by:
- Adding new equation implementations in the `problem/` directory
- Modifying neural network architectures in the `PINN` class
- Implementing additional sampling or visualization techniques

## Troubleshooting

- Ensure all dependencies are correctly installed
- Check that your TensorFlow version is compatible with the PINN implementation
- Verify input ranges and sampling methods match your specific use case

## Future Work

- Implement more complex fluid dynamics scenarios
- Enhance visualization techniques
- Add support for more diverse boundary conditions

## License

This project is licensed under the MIT License. See the LICENSE file for details.
