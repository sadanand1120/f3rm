# CMA-ES Optimization Visualization

This directory contains tools for running CMA-ES optimization and creating animated visualizations similar to those shown in [David Ha's blog post](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/).

## Files

- `parallel_cmaes.py` - Main CMA-ES optimization engine with history recording
- `visualize_cmaes.py` - Visualization script for creating animated plots
- `demo_cmaes_visualization.py` - Demo script showing examples

## Features

- **Parallel CMA-ES optimization** using multiple workers and GPU support
- **History recording** of samples, means, and best solutions across generations
- **Animated visualizations** showing optimization progress on 2D functions
- **Multiple test functions**: Rastrigin, Schaffer, and simple quadratic functions
- **Export options**: GIF, MP4, or individual frames

## Quick Start

### 1. Run the Demo

```bash
python demo_cmaes_visualization.py
```

This will:
- Run CMA-ES on Rastrigin, Schaffer, and toy functions
- Save optimization history to JSON files
- Create animated GIF visualizations
- Show one interactive plot

### 2. Run Individual Optimization

```python
from parallel_cmaes import cma_es_optimize, RastriginFactory

# Create objective function factory (2D Rastrigin with no shift)
objective = RastriginFactory(shift=0.0, enforce_2d=True, use_fixed_constant=20)

# Run optimization with history recording
best_x, best_f = cma_es_optimize(
    obj_source=objective,
    x0=np.array([3.0, -2.0]),
    sigma0=1.0,
    lower_bounds=np.array([-5.12, -5.12]),
    upper_bounds=np.array([5.12, 5.12]),
    popsize=40,
    max_epochs=30,
    repeats=1,
    n_workers=4,
    record_history=True,
    history_file="my_optimization.json"
)
```

### 3. Create Visualizations

```bash
# Interactive visualization
python visualize_cmaes.py optimization_history.json

# Save as GIF
python visualize_cmaes.py optimization_history.json --output animation.gif

# Save as MP4 (requires ffmpeg)
python visualize_cmaes.py optimization_history.json --output animation.mp4

# Save individual frames
python visualize_cmaes.py optimization_history.json --save-frames
```

## Visualization Features

The animated plots show:

- **Background contours**: Objective function landscape
- **Blue dots**: Current generation samples
- **Green dot**: Mean of the current distribution
- **Red dot**: Best solution found so far
- **Title info**: Generation number, best fitness, and sigma value

## Supported Objective Functions

### RastriginFactory
- Classic multimodal test function with many local optima
- Configurable parameters:
  - `shift`: Offset for the function (default: 10.0 for general case, 0.0 for 2D visualization)
  - `enforce_2d`: Whether to enforce 2D constraint (default: False)
  - `use_fixed_constant`: Use fixed constant instead of 10*len(z) (default: None, use 20 for 2D visualization)
- Global minimum depends on shift parameter
- Examples:
  - General N-dimensional: `RastriginFactory(shift=10.0)` - minimum at (10, 10, ...)
  - 2D visualization: `RastriginFactory(shift=0.0, enforce_2d=True, use_fixed_constant=20)` - minimum at (0, 0)

### SchafferFactory  
- Another multimodal function with circular symmetry
- Global minimum at (0, 0) with value 0

### ToyFactory
- Simple quadratic function: (x-offset)² + (y-offset)²
- Configurable offset parameter
- Good for testing basic optimization behavior

### DiscreteCircleFactory
- Discrete step function: value 0 inside a small circle, 1 everywhere else
- Demonstrates CMA-ES failure on discontinuous functions
- Configurable center position and radius
- Useful for showing optimization limitations

### NERFOpt
- Real-world 3D pose optimization for NeRF-based robotics applications
- Optimizes camera position (px, py) and rotation (ry) parameters
- Visualization shows first 2 dimensions (px, py) with rotation fixed to 0
- Demonstrates CMA-ES on complex, real-world objective functions
- Requires NeRF model and related dependencies

## Dependencies

```bash
pip install numpy matplotlib torch cma tqdm
```

For MP4 export, you'll also need:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## Customization

### Adding New Objective Functions

1. Create a factory class in `parallel_cmaes.py`:

```python
class MyFunctionFactory:
    def __init__(self, **kwargs):
        self.param = kwargs.get('param', 1.0)
    
    def __call__(self, device: torch.device):
        def my_function(x: np.ndarray):
            # Your objective function here
            return np.sum(x**2) * self.param
        return my_function
```

2. Add the visualization function in `visualize_cmaes.py`:

```python
def create_objective_function(objective_name):
    # ... existing code ...
    elif objective_name == "MyFunctionFactory":
        def my_viz_function(x, y):
            return (x**2 + y**2) * self.param  # Adjust as needed
        return my_viz_function
```

### Adjusting Visualization

- Modify `resolution` in `create_contour_plot()` for finer/coarser contours
- Change `interval` in `FuncAnimation()` for faster/slower animation
- Adjust `markersize` and `alpha` for different dot appearances
- Modify `levels` in contour plot for different contour spacing

## Examples

See the `demo_cmaes_visualization.py` for complete examples of:
- Setting up different objective functions
- Configuring optimization parameters
- Creating and saving visualizations

## Tips

- Use smaller population sizes (20-50) for clearer visualizations
- Limit to 20-50 generations for reasonable animation length
- Start with sigma0 ≈ (search_range / 3) for good initial spread
- Use fewer workers (2-4) when recording history to reduce overhead 