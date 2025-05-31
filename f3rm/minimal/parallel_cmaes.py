# Standard library imports
import inspect
import json
import os
import sys
import time
from typing import Callable, Optional, Sequence

# Third-party imports
import cma
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from tqdm.auto import tqdm

# Set the multiprocessing start method to 'spawn' for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Start method already set


def convert_numpy_for_json(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_for_json(item) for item in obj]
    else:
        return obj


def _worker(param_q: Queue, result_q: Queue, stop_q: Queue, obj_source, worker_device: torch.device):
    """Worker process that evaluates objectives on specified device."""
    obj_fn = obj_source(worker_device)   # Build objective function from factory

    # Redirect worker output to avoid clutter
    pid = str(os.getpid())
    sys.stdout = open(f"/tmp/cma_{pid}.out", "a")
    sys.stderr = open(f"/tmp/cma_{pid}.err", "a")

    # Main evaluation loop
    while stop_q.empty():
        if param_q.empty():
            time.sleep(0.01)
            continue
        sid, x = param_q.get()
        fitness = obj_fn(x)
        result_q.put((sid, fitness))


class ParallelEvaluator:
    """Parallel objective function evaluator using multiple workers."""

    def __init__(self, obj_source, n_workers: int = 8, show_progress: bool = True):
        self.param_q, self.result_q, self.stop_q = Queue(), Queue(), Queue()
        self.show_progress = show_progress

        # Create workers with device assignment
        self.workers = []
        for wid in range(n_workers):
            worker_device = torch.device(f"cuda:{wid % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
            worker = Process(target=_worker, args=(self.param_q, self.result_q, self.stop_q, obj_source, worker_device))
            self.workers.append(worker)
            worker.start()

    def evaluate(self, params: Sequence[np.ndarray], repeats: int = 1) -> list[float]:
        """Evaluate parameters and return mean fitness scores (lower = better)."""
        n_tasks = len(params) * repeats
        agg = [0.0] * len(params)

        # Submit all tasks
        for sid, p in enumerate(params):
            for _ in range(repeats):
                self.param_q.put((sid, p))

        # Collect results
        bar = tqdm(total=n_tasks, disable=not self.show_progress, desc="Evaluations", leave=False)
        processed = 0
        while processed < n_tasks:
            sid, f = self.result_q.get()
            agg[sid] += f / repeats
            processed += 1
            bar.update(1)
        bar.close()
        return agg

    def close(self, grace: float = 2.0):
        """Shutdown workers gracefully, then force-kill if needed."""
        # Signal stop
        self.stop_q.put(True)

        # Wait for graceful exit
        for w in self.workers:
            w.join(timeout=grace)

        # Force kill stragglers
        for w in self.workers:
            if w.is_alive():
                print(f"[ParallelEvaluator] Worker {w.pid} still alive → terminate()")
                w.terminate()
                w.join()

        # Clean up log files
        for w in self.workers:
            for ext in (".out", ".err"):
                try:
                    os.remove(f"/tmp/cma_{w.pid}{ext}")
                except (FileNotFoundError, PermissionError):
                    pass

        # Close queues
        for q in (self.param_q, self.result_q, self.stop_q):
            q.close()
            q.join_thread()


def cma_es_optimize(obj_source: Callable,
                    x0: np.ndarray,
                    sigma0: float = 0.5,
                    lower_bounds: np.ndarray = None,
                    upper_bounds: np.ndarray = None,
                    popsize: int = 64,
                    max_epochs: int = 1000,
                    repeats: int = 4,
                    target: float = None,
                    n_workers: int = 8,
                    record_history: bool = False,
                    history_file: str = None,
                    enable_stop: bool = False,
                    seed: int = 42):
    """Parallel CMA-ES optimization. Returns (best_params, best_fitness)."""
    # Set random seed for reproducibility
    np.random.seed(seed)

    evaluator = ParallelEvaluator(obj_source, n_workers=n_workers)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': popsize,
        'verb_disp': 0,
        'bounds': [lower_bounds, upper_bounds],
        'seed': seed
    })
    best_params, best_fit = None, float("inf")

    # History recording
    history = {
        'generations': [],
        'objective_name': obj_source.__class__.__name__,
        'x0': x0.tolist(),
        'sigma0': sigma0,
        'popsize': popsize,
        'bounds': [lower_bounds.tolist() if lower_bounds is not None else None,
                   upper_bounds.tolist() if upper_bounds is not None else None],
        'factory_params': convert_numpy_for_json(getattr(obj_source, '__dict__', {}))  # Store factory parameters
    } if record_history else None

    try:
        with tqdm(range(max_epochs), desc="Epochs") as bar:
            for epoch in bar:
                if enable_stop and es.stop():
                    break
                sols = es.ask()
                fitnesses = evaluator.evaluate(sols, repeats=repeats)
                es.tell(sols, fitnesses)
                idx = int(np.argmin(fitnesses))
                gen_best = fitnesses[idx]
                if gen_best < best_fit:
                    best_fit, best_params = gen_best, sols[idx]

                # Record history for visualization
                if record_history:
                    generation_data = {
                        'epoch': epoch,
                        'samples': [sol.tolist() for sol in sols],
                        'fitnesses': fitnesses,
                        'mean': es.mean.tolist(),
                        'sigma': float(es.sigma),
                        'best_solution': best_params.tolist() if best_params is not None else None,
                        'best_fitness': float(best_fit)
                    }
                    history['generations'].append(generation_data)

                bar.update(0)   # refresh only, iteration already counted
                bar.set_postfix(best=f"{best_fit:.4g}")
                if target is not None and best_fit <= target:
                    break
    finally:
        print("Closing evaluator...")
        evaluator.close()

        # Save history if requested
        if record_history and history_file:
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"History saved to {history_file}")

    return best_params, best_fit


# Example objective functions
class RastriginFactory:
    """Factory for creating Rastrigin objective functions.

    Can create both general N-dimensional and specific 2D versions based on parameters.
    """

    def __init__(self, **kwargs):
        self.shift = kwargs.get('shift', 10.0)  # Default shift for general case
        self.enforce_2d = kwargs.get('enforce_2d', False)  # Whether to enforce 2D constraint
        self.use_fixed_constant = kwargs.get('use_fixed_constant', None)  # Use fixed constant instead of 10*len(z)

        # Visualization optimization parameters
        self.viz_opt_params = {
            'x0': np.array([4.0, -4.0]),
            'sigma0': 1.0,
            'lower_bounds': np.array([-5.12, -5.12]),
            'upper_bounds': np.array([5.12, 5.12]),
            'popsize': 40,
            'max_epochs': 30,
            'repeats': 1,
            'n_workers': 1
        }

    def __call__(self, device: torch.device):
        def rastrigin(x: np.ndarray):
            """Rastrigin function with configurable shift and dimensionality."""
            if self.enforce_2d:
                assert len(x) == 2, "This Rastrigin function is configured for 2D only"
            z = x - self.shift
            # Use fixed constant if specified, otherwise use 10*len(z)
            if self.use_fixed_constant is not None:
                constant = self.use_fixed_constant
            else:
                constant = 10 * len(z)

            return constant + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))
        return rastrigin


class SchafferFactory:
    """Factory for creating Schaffer-2D objective functions."""

    def __init__(self, **kwargs):
        # Visualization optimization parameters
        self.viz_opt_params = {
            'x0': np.array([50.0, 50.0]),
            'sigma0': 3.0,
            'lower_bounds': np.array([-100.0, -100.0]),
            'upper_bounds': np.array([100.0, 100.0]),
            'popsize': 40,
            'max_epochs': 25,
            'repeats': 1,
            'n_workers': 1
        }

    def __call__(self, device: torch.device):
        def schaffer(x: np.ndarray):
            """Schaffer function for 2D optimization."""
            assert len(x) == 2, "Schaffer function is designed for 2D"
            x1, x2 = x[0], x[1]
            numerator = np.sin(x1**2 + x2**2)**2 - 0.5
            denominator = (1 + 0.001 * (x1**2 + x2**2))**2
            return 0.5 + numerator / denominator
        return schaffer


class ToyFactory:
    """Factory for creating Toy objective functions."""

    def __init__(self, **kwargs):
        self.offset = kwargs.get('offset', 3.0)

        # Visualization optimization parameters
        self.viz_opt_params = {
            'x0': np.array([-2.0, -4.0]),
            'sigma0': 0.2,
            'lower_bounds': np.array([-5.0, -5.0]),
            'upper_bounds': np.array([5.0, 5.0]),
            'popsize': 30,
            'max_epochs': 20,
            'repeats': 1,
            'n_workers': 1
        }

    def __call__(self, device: torch.device):
        def obj(x: np.ndarray):
            return np.sum((x - self.offset)**2)
        return obj


class DiscreteCircleFactory:
    """Factory for creating discrete circle objective functions.

    Function is 1 inside a circle of given radius around center, 0 elsewhere.
    This demonstrates CMA-ES failure on discrete/discontinuous functions.
    """

    def __init__(self, **kwargs):
        self.center_x = kwargs.get('center_x', 2.0)
        self.center_y = kwargs.get('center_y', 2.0)
        self.radius = kwargs.get('radius', 0.2)

        # Visualization optimization parameters
        self.viz_opt_params = {
            'x0': np.array([-2.0, -2.0]),
            'sigma0': 0.5,
            'lower_bounds': np.array([-3.0, -3.0]),
            'upper_bounds': np.array([5.0, 5.0]),
            'popsize': 40,
            'max_epochs': 30,
            'repeats': 1,
            'n_workers': 1
        }

    def __call__(self, device: torch.device):
        def obj(x: np.ndarray):
            """Discrete circle function - 1 inside circle, 0 outside."""
            assert len(x) == 2, "DiscreteCircle function is designed for 2D"
            dist = np.sqrt((x[0] - self.center_x)**2 + (x[1] - self.center_y)**2)
            return 0.0 if dist <= self.radius else 1.0
        return obj


class HeavyFactory:
    """Factory for creating Heavy objective functions that use GPU."""

    def __init__(self, **kwargs):
        self.model_size = kwargs.get('model_size', 1000)

        # Visualization optimization parameters
        self.viz_opt_params = {
            'x0': np.array([0.0, 0.0]),
            'sigma0': 0.5,
            'lower_bounds': np.array([-2.0, -2.0]),
            'upper_bounds': np.array([2.0, 2.0]),
            'popsize': 30,
            'max_epochs': 20,
            'repeats': 1,
            'n_workers': 1
        }

    def __call__(self, device: torch.device):
        # Create model on the specified device
        model = torch.randn(self.model_size, self.model_size, device=device)

        def obj(x: np.ndarray):
            t = torch.as_tensor(x, dtype=torch.float32, device=device)
            return float(((t @ t) + model[0, 0]).cpu())
        return obj


if __name__ == "__main__":
    # Example for 2D optimization with visualization
    rastrigin_2d_factory = RastriginFactory(shift=0.0, enforce_2d=True, use_fixed_constant=20)
    # rastrigin_factory = RastriginFactory(shift=10.0)
    # schaffer_factory = SchafferFactory()
    # heavy_factory = HeavyFactory(model_size=50000)
    # toy_factory = ToyFactory(offset=3.0)

    best_x, best_f = cma_es_optimize(
        # obj_source=heavy_factory,
        # x0=np.zeros(100),
        # sigma0=0.5,
        # popsize=101,
        # max_epochs=1000,
        obj_source=rastrigin_2d_factory,
        x0=np.array([3.0, -2.0]),  # Start away from optimum
        sigma0=1.0,
        lower_bounds=np.array([-5.0, -5.0]),
        upper_bounds=np.array([5.0, 5.0]),
        popsize=50,
        max_epochs=50,
        repeats=1,
        n_workers=4,
        target=None,
        record_history=True,
        history_file="optimization_history.json",
        seed=42
    )

    print(f"Best fitness: {best_f}")
    print(f"Best params: {best_x}")
    print("History saved to optimization_history.json")
