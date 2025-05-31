#!/usr/bin/env python3
"""
Demo script showing CMA-ES optimization with visualization.
Runs optimization on 2D functions and creates animated plots.
"""

import numpy as np
import os


def demo_rastrigin():
    """Demo CMA-ES on 2D Rastrigin function."""
    from parallel_cmaes import cma_es_optimize, RastriginFactory

    print("Running CMA-ES on 2D Rastrigin function...")

    rastrigin_factory = RastriginFactory(shift=0.0, enforce_2d=True, use_fixed_constant=20)

    best_x, best_f = cma_es_optimize(
        obj_source=rastrigin_factory,
        x0=np.array([4.0, -4.0]),  # Start away from optimum
        sigma0=1.0,
        lower_bounds=np.array([-5.12, -5.12]),
        upper_bounds=np.array([5.12, 5.12]),
        popsize=40,
        max_epochs=30,
        repeats=1,
        n_workers=1,  # Single worker to avoid multiprocessing issues
        target=None,
        record_history=True,
        history_file="rastrigin_history.json",
        enable_stop=False  # Keep CMA-ES stop criteria disabled
    )

    print(f"Rastrigin - Best fitness: {best_f:.4f}")
    print(f"Rastrigin - Best params: {best_x}")
    print("History saved to rastrigin_history.json")

    return "rastrigin_history.json"


def demo_schaffer():
    """Demo CMA-ES on 2D Schaffer function."""
    from parallel_cmaes import cma_es_optimize, SchafferFactory

    print("\nRunning CMA-ES on 2D Schaffer function...")

    schaffer_factory = SchafferFactory()

    best_x, best_f = cma_es_optimize(
        obj_source=schaffer_factory,
        x0=np.array([50.0, 50.0]),  # Start away from optimum
        sigma0=3.0,
        lower_bounds=np.array([-100.0, -100.0]),
        upper_bounds=np.array([100.0, 100.0]),
        popsize=40,
        max_epochs=25,
        repeats=1,
        n_workers=1,  # Single worker to avoid multiprocessing issues
        target=None,
        record_history=True,
        history_file="schaffer_history.json"
    )

    print(f"Schaffer - Best fitness: {best_f:.4f}")
    print(f"Schaffer - Best params: {best_x}")
    print("History saved to schaffer_history.json")

    return "schaffer_history.json"


def demo_toy():
    """Demo CMA-ES on simple toy function."""
    from parallel_cmaes import cma_es_optimize, ToyFactory

    print("\nRunning CMA-ES on toy quadratic function...")

    toy_factory = ToyFactory(offset=4.0)

    best_x, best_f = cma_es_optimize(
        obj_source=toy_factory,
        x0=np.array([-2.0, -4.0]),
        sigma0=0.2,
        lower_bounds=np.array([-5.0, -5.0]),
        upper_bounds=np.array([5.0, 5.0]),
        popsize=30,
        max_epochs=20,
        repeats=1,
        n_workers=1,  # Single worker to avoid multiprocessing issues
        target=None,
        record_history=True,
        history_file="toy_history.json"
    )

    print(f"Toy - Best fitness: {best_f:.4f}")
    print(f"Toy - Best params: {best_x}")
    print("History saved to toy_history.json")

    return "toy_history.json"


def demo_discrete_circle():
    """Demo CMA-ES on discrete circle function (shows CMA-ES failure)."""
    from parallel_cmaes import cma_es_optimize, DiscreteCircleFactory

    print("\nRunning CMA-ES on discrete circle function (demonstrating failure)...")

    discrete_factory = DiscreteCircleFactory(center_x=2.0, center_y=2.0, radius=0.2)

    best_x, best_f = cma_es_optimize(
        obj_source=discrete_factory,
        x0=np.array([-2.0, -2.0]),  # Start far away from target
        sigma0=0.5,
        lower_bounds=np.array([-3.0, -3.0]),
        upper_bounds=np.array([5.0, 5.0]),
        popsize=40,
        max_epochs=30,
        repeats=1,
        n_workers=1,  # Single worker to avoid multiprocessing issues
        target=None,
        record_history=True,
        history_file="discrete_circle_history.json"
    )

    print(f"Discrete Circle - Best fitness: {best_f:.4f}")
    print(f"Discrete Circle - Best params: {best_x}")
    print(f"Target is inside circle at (2.0, 2.0) with radius 0.2")
    print("This demonstrates CMA-ES difficulty with discrete functions!")
    print("History saved to discrete_circle_history.json")

    return "discrete_circle_history.json"


def demo_nerf_opt():
    """Demo CMA-ES on NERFOpt function (3D optimization, 2D visualization)."""
    from parallel_cmaes import cma_es_optimize
    from opt import NERFOpt

    print("\nRunning CMA-ES on NERFOpt function (3D pose optimization)...")

    nerf_factory = NERFOpt()

    best_x, best_f = cma_es_optimize(
        obj_source=nerf_factory,
        x0=np.array([0.0, 0.0, 0.0]),  # px, py, ry (in radians / 10)
        sigma0=0.3,
        lower_bounds=np.array([-1.0, -1.0, -0.5]),
        upper_bounds=np.array([1.0, 1.0, 0.5]),
        popsize=50,
        max_epochs=20,
        repeats=1,
        n_workers=1,  # Single worker to avoid multiprocessing issues
        target=None,
        record_history=True,
        history_file="nerf_opt_history.json"
    )

    print(f"NERFOpt - Best fitness: {best_f:.4f}")
    print(f"NERFOpt - Best params: {best_x}")
    print("This demonstrates CMA-ES on a real 3D optimization problem!")
    print("Visualization shows first 2 dimensions (px, py) with no background")
    print("History saved to nerf_opt_history.json")

    return "nerf_opt_history.json"


def main():
    from visualize_cmaes import visualize_optimization

    print("CMA-ES Visualization Demo")
    print("=" * 50)

    # Run optimizations
    history_files = []

    try:
        history_files.append(demo_rastrigin())
    except Exception as e:
        print(f"Error running Rastrigin demo: {e}")

    try:
        history_files.append(demo_schaffer())
    except Exception as e:
        print(f"Error running Schaffer demo: {e}")

    try:
        history_files.append(demo_toy())
    except Exception as e:
        print(f"Error running Toy demo: {e}")

    try:
        history_files.append(demo_discrete_circle())
    except Exception as e:
        print(f"Error running Discrete Circle demo: {e}")

    try:
        history_files.append(demo_nerf_opt())
    except Exception as e:
        print(f"Error running NERFOpt demo: {e}")

    print("\n" + "=" * 50)
    print("Optimization completed! Now creating visualizations...")

    # Create visualizations
    for i, history_file in enumerate(history_files):
        if os.path.exists(history_file):
            print(f"\nVisualizing {history_file}...")
            try:
                # Create both interactive and saved animations
                function_name = history_file.split('_')[0]
                gif_file = f"{function_name}_optimization.gif"

                fig, anim = visualize_optimization(
                    history_file,
                    output_file=gif_file,
                    show_animation=(i == 0)  # Only show first one interactively
                )

                print(f"Animation saved as {gif_file}")

            except Exception as e:
                print(f"Error creating visualization for {history_file}: {e}")

    print("\nDemo completed!")
    print("You can view the individual animations by running:")
    for history_file in history_files:
        if os.path.exists(history_file):
            print(f"  python visualize_cmaes.py {history_file}")


if __name__ == "__main__":
    main()
