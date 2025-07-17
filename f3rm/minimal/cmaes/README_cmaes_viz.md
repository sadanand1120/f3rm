# CMA-ES Optimization Visualization

Simple tools for running CMA-ES optimization and creating animated visualizations.

## Files

- `parallel_cmaes.py` - CMA-ES optimization engine with factory classes
- `visualize_cmaes.py` - Visualization script for animated plots  
- `demo_cmaes_visualization.py` - Simple demo script

## Quick Start

### Run the Demo

```bash
python demo_cmaes_visualization.py
```

This runs CMA-ES on a chosen factory class and creates an animated visualization.

### Switch Between Functions

Just change one line in `demo_cmaes_visualization.py`:

```python
# Choose any factory class:
factory = RastriginFactory(shift=0.0, enforce_2d=True, use_fixed_constant=20)
# factory = SchafferFactory()
# factory = ToyFactory(offset=4.0)
# factory = DiscreteCircleFactory(center_x=2.0, center_y=2.0, radius=0.2)
```

### Available Factory Classes

Each factory has built-in `viz_opt_params` for plug-and-play visualization:

- **RastriginFactory** - Multimodal function with many local optima
- **SchafferFactory** - Circular multimodal function  
- **ToyFactory** - Simple quadratic function
- **DiscreteCircleFactory** - Discrete step function (shows CMA-ES failure)
- **HeavyFactory** - GPU-based function
- **NERFOpt** - Real 3D pose optimization (requires NeRF dependencies)

### Custom Usage

```python
from f3rm.minimal.parallel_cmaes import cma_es_optimize, RastriginFactory
from f3rm.minimal.visualize_cmaes import visualize_optimization

# Create factory
factory = RastriginFactory(shift=0.0, enforce_2d=True, use_fixed_constant=20)

# Run optimization using factory's built-in parameters
params = factory.viz_opt_params
best_x, best_f = cma_es_optimize(
    obj_source=factory,
    x0=params['x0'],
    sigma0=params['sigma0'],
    lower_bounds=params['lower_bounds'],
    upper_bounds=params['upper_bounds'],
    popsize=params['popsize'],
    max_epochs=params['max_epochs'],
    repeats=params['repeats'],
    n_workers=params['n_workers'],
    record_history=True,
    history_file="my_history.json"
)

# Create visualization
visualize_optimization("my_history.json", output_file="my_animation.gif")
```

### Visualization Only

```bash
python visualize_cmaes.py history_file.json --output animation.gif
```

## Dependencies

```bash
pip install numpy matplotlib torch cma tqdm
```

For MP4 export: `sudo apt install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS) 