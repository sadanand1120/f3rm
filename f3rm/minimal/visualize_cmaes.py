#!/usr/bin/env python3
"""
Visualizer for CMA-ES optimization history.
Creates animated plots similar to those in: https://blog.otoro.net/2017/10/29/visual-evolution-strategies/
"""

# Standard library imports
import argparse
import json
import sys
import os

# Third-party imports
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

# Repo-specific imports
from f3rm.minimal.opt import NERFOpt
from f3rm.minimal.parallel_cmaes import (SchafferFactory, ToyFactory,
                                         RastriginFactory, HeavyFactory, DiscreteCircleFactory)


def create_objective_function(objective_name, **kwargs):
    """Create the objective function based on the factory name."""
    # Use CUDA as default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Map factory names to factory classes
    factory_map = {
        'Rastrigin2DFactory': lambda **kw: RastriginFactory(shift=0.0, enforce_2d=True, use_fixed_constant=20, **kw),
        'SchafferFactory': SchafferFactory,
        'ToyFactory': ToyFactory,
        'RastriginFactory': RastriginFactory,
        'HeavyFactory': HeavyFactory,
        'DiscreteCircleFactory': DiscreteCircleFactory,
        'NERFOpt': NERFOpt,
    }

    # Create the factory and get the actual objective function
    factory_class = factory_map[objective_name]
    factory = factory_class(**kwargs)
    obj_fn = factory(device)

    # For NERFOpt, create a lambda that converts 2D input to 3D with z=0
    if objective_name == 'NERFOpt':
        base_fn = obj_fn
        def obj_fn(x_arr): return base_fn(np.append(x_arr, 0.0) if len(x_arr) == 2 else x_arr)

    # Wrap the function to work with 2D meshgrids
    def wrapped_fn(x, y):
        if hasattr(x, 'shape') and len(x.shape) > 0:
            # Meshgrid case - x and y are 2D arrays
            result = np.zeros_like(x)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    result[i, j] = obj_fn(np.array([x[i, j], y[i, j]]))
            return result
        else:
            # Single point case
            return obj_fn(np.array([x, y]))

    return wrapped_fn


def create_contour_plot(objective_fn, bounds, resolution=100):
    """Create contour plot data for the objective function."""
    x_min, x_max = bounds[0][0], bounds[1][0]
    y_min, y_max = bounds[0][1], bounds[1][1]

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = objective_fn(X, Y)

    return X, Y, Z


def visualize_optimization(history_file, output_file=None, show_animation=True, save_frames=False, linear_scale=False):
    """Create animated visualization of CMA-ES optimization."""

    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)

    # Extract data
    generations = history['generations']
    objective_name = history['objective_name']
    bounds = history['bounds']

    # Get factory parameters if they exist in history
    factory_params = history.get('factory_params', {})

    if bounds[0] is None or bounds[1] is None:
        print("Warning: No bounds specified in history. Using default bounds.")
        bounds = [[-1.25, -0.75], [1.75, 1.75]]

    # Create objective function using the actual factory
    objective_fn = create_objective_function(objective_name, **factory_params)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create contour plot if we have an objective function
    if objective_fn is not None:
        # Create contour plot
        X, Y, Z = create_contour_plot(objective_fn, bounds)

        # Handle infinite values for all objective functions
        Z_finite = np.where(np.isfinite(Z), Z, np.nanmax(Z[np.isfinite(Z)]) * 2 if np.any(np.isfinite(Z)) else 0)

        if linear_scale:
            # Use linear normalization
            levels = np.linspace(Z_finite.min(), Z_finite.max(), 100)
            contour = ax.contourf(X, Y, Z_finite, levels=levels, cmap='viridis', alpha=0.8)
            ax.contour(X, Y, Z_finite, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        else:
            # Use logarithmic normalization (ensure positive values)
            Z_positive = Z_finite - Z_finite.min() + 1e-10
            levels = np.logspace(np.log10(Z_positive.min()), np.log10(Z_positive.max()), 100)
            contour = ax.contourf(X, Y, Z_positive, levels=levels, norm=LogNorm(), cmap='viridis', alpha=0.8)
            ax.contour(X, Y, Z_positive, levels=levels, norm=LogNorm(), colors='black', alpha=0.3, linewidths=0.5)

        # Add colorbar
        plt.colorbar(contour, ax=ax, label='Objective Value')
    else:
        # No background - just set up the plot area
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.3)

    # Initialize plot elements
    samples_plot, = ax.plot([], [], 'bo', markersize=2, alpha=0.6, label='Samples')
    best_plot, = ax.plot([], [], 'go', markersize=8, label='Best')
    mean_plot, = ax.plot([], [], 'ro', markersize=8, label='Mean')

    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'CMA-ES Optimization: {objective_name}')
    ax.legend()

    # Set axis limits
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])

    def animate(frame):
        """Animation function."""
        if frame >= len(generations):
            return samples_plot, mean_plot, best_plot

        gen = generations[frame]

        # Extract samples (assuming 2D, but handle 3D by taking first 2 dims)
        samples = np.array(gen['samples'])
        mean = np.array(gen['mean'])
        best_solution = np.array(gen['best_solution']) if gen['best_solution'] else mean

        # Handle 3D data by taking only first 2 dimensions for visualization
        if samples.shape[1] > 2:
            samples = samples[:, :2]
        if len(mean) > 2:
            mean = mean[:2]
        if len(best_solution) > 2:
            best_solution = best_solution[:2]

        # Update plots
        samples_plot.set_data(samples[:, 0], samples[:, 1])
        mean_plot.set_data([mean[0]], [mean[1]])
        best_plot.set_data([best_solution[0]], [best_solution[1]])

        # Update title with generation info
        ax.set_title(f'CMA-ES Optimization: {objective_name} | '
                     f'Generation: {gen["epoch"]} | '
                     f'Best Fitness: {gen["best_fitness"]:.4f} | '
                     f'Sigma: {gen["sigma"]:.4f}')

        return samples_plot, mean_plot, best_plot

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(generations),
        interval=200, blit=False, repeat=True
    )

    # Save frames if requested
    if save_frames:
        frame_dir = "cmaes_frames"
        os.makedirs(frame_dir, exist_ok=True)

        for i, gen in enumerate(generations):
            animate(i)
            plt.savefig(f"{frame_dir}/frame_{i:03d}.png", dpi=150, bbox_inches='tight')
            print(f"Saved frame {i+1}/{len(generations)}")

    # Save as GIF if output file specified
    if output_file:
        print(f"Saving animation to {output_file}...")
        if output_file.endswith('.gif'):
            anim.save(output_file, writer='pillow', fps=5)
        elif output_file.endswith('.mp4'):
            anim.save(output_file, writer='ffmpeg', fps=5)
        else:
            anim.save(output_file + '.gif', writer='pillow', fps=5)
        print("Animation saved!")

    # Show animation
    if show_animation:
        plt.tight_layout()
        plt.show()

    return fig, anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize CMA-ES optimization history')
    parser.add_argument('history_file', help='JSON file containing optimization history')
    parser.add_argument('--output', '-o', help='Output file for animation (gif/mp4)')
    parser.add_argument('--no-show', action='store_true', help='Don\'t show interactive animation')
    parser.add_argument('--save-frames', action='store_true', help='Save individual frames')
    parser.add_argument('--linear-scale', action='store_true', help='Use linear scale for contour plot')

    args = parser.parse_args()

    visualize_optimization(
        args.history_file,
        output_file=args.output,
        show_animation=not args.no_show,
        save_frames=args.save_frames,
        linear_scale=args.linear_scale
    )
