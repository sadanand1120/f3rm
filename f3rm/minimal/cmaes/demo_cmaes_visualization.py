#!/usr/bin/env python3
"""
Demo script showing CMA-ES optimization with visualization.
Simple example that runs optimization on one factory and creates animated plot.
"""

# Third-party imports
import numpy as np

# Repo-specific imports
from f3rm.minimal.cmaes.parallel_cmaes import (cma_es_optimize, RastriginFactory, SchafferFactory,
                                               ToyFactory, DiscreteCircleFactory)
from f3rm.minimal.cmaes.visualize_cmaes import visualize_optimization

if __name__ == "__main__":
    # Choose factory class to run - just change this line to try different functions!
    # factory = RastriginFactory(shift=0.0, enforce_2d=True, use_fixed_constant=20)
    # factory = SchafferFactory()
    factory = ToyFactory(offset=4.0)
    # factory = DiscreteCircleFactory(center_x=2.0, center_y=2.0, radius=0.2)

    # Run optimization using factory's viz_opt_params
    params = factory.viz_opt_params
    history_file = "demo_history.json"

    print(f"Running CMA-ES on {factory.__class__.__name__}...")

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
        history_file=history_file
    )

    print(f"Best fitness: {best_f:.4f}")
    print(f"Best params: {best_x}")
    print(f"History saved to {history_file}")

    # Create visualization
    print("Creating visualization...")
    visualize_optimization(
        history_file,
        output_file="demo_optimization.gif",
        show_animation=True
    )
    print("Demo completed!")
