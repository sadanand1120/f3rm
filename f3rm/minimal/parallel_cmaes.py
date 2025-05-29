# parallel_cma.py
import os
import sys
import time
import torch
import torch.multiprocessing as mp
import numpy as np
from torch.multiprocessing import Process, Queue
import cma
from typing import Callable, Sequence, Optional
from tqdm.auto import tqdm
import inspect

# Set the multiprocessing start method to 'spawn' for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Start method already set


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

    def close(self, grace: float = 5.0):
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
                    n_workers: int = 8):
    """Parallel CMA-ES optimization. Returns (best_params, best_fitness)."""
    evaluator = ParallelEvaluator(obj_source, n_workers=n_workers)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': popsize, 'verb_disp': 0, 'bounds': [lower_bounds, upper_bounds]})
    best_params, best_fit = None, float("inf")
    try:
        with tqdm(range(max_epochs), desc="Epochs") as bar:
            for _ in bar:
                if es.stop():
                    break
                sols = es.ask()
                fitnesses = evaluator.evaluate(sols, repeats=repeats)
                es.tell(sols, fitnesses)
                idx = int(np.argmin(fitnesses))
                gen_best = fitnesses[idx]
                if gen_best < best_fit:
                    best_fit, best_params = gen_best, sols[idx]
                bar.update(0)   # refresh only, iteration already counted
                bar.set_postfix(best=f"{best_fit:.4g}")
                if target is not None and best_fit <= target:
                    break
    finally:
        print("Closing evaluator...")
        evaluator.close()
    return best_params, best_fit


# Example objective functions
class RastriginFactory:
    """Factory for creating Rastrigin objective functions."""

    def __init__(self, **kwargs):
        pass  # No additional parameters needed for rastrigin

    def __call__(self, device: torch.device):
        def rastrigin(x: np.ndarray):
            """Rastrigin function (shifted + negated for minimization)."""
            z = x - 10.0
            return 10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))
        return rastrigin


class ToyFactory:
    """Factory for creating Toy objective functions."""

    def __init__(self, **kwargs):
        self.offset = kwargs.get('offset', 3.0)

    def __call__(self, device: torch.device):
        def obj(x: np.ndarray):
            return np.sum((x - self.offset)**2)
        return obj


class HeavyFactory:
    """Factory for creating Heavy objective functions that use GPU."""

    def __init__(self, **kwargs):
        self.model_size = kwargs.get('model_size', 1000)

    def __call__(self, device: torch.device):
        # Create model on the specified device
        model = torch.randn(self.model_size, self.model_size, device=device)

        def obj(x: np.ndarray):
            t = torch.as_tensor(x, dtype=torch.float32, device=device)
            return float(((t @ t) + model[0, 0]).cpu())
        return obj


if __name__ == "__main__":
    # Example
    heavy_factory = HeavyFactory(model_size=50000)
    # toy_factory = ToyFactory(offset=3.0)
    # rastrigin_factory = RastriginFactory()

    best_x, best_f = cma_es_optimize(
        obj_source=heavy_factory,
        x0=np.zeros(100),
        sigma0=0.5,
        popsize=101,
        max_epochs=1000,
        repeats=1,
        n_workers=4,
        target=None
    )

    print(f"Best fitness: {best_f}")
    print(f"Best params (first 6 dims): {best_x[:6]}")
