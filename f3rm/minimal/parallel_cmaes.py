# parallel_cma.py -------------------------------------------------------------
import os
import sys
import time
import torch
import numpy as np
from torch.multiprocessing import Process, Queue
import cma                                   # pip install cma
from typing import Callable, Sequence
from tqdm.auto import tqdm
import inspect


def _worker(param_q: Queue, result_q: Queue, stop_q: Queue, obj_source, w_id: int):
    """GPU-aware worker that builds (if needed) its own objective fn."""
    dev = torch.device(f"cuda:{w_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")

    # pick or build objective -------------------------------------------------
    if callable(obj_source):
        try:                                # zero-arg factory?
            if len(inspect.signature(obj_source).parameters) == 1:
                obj_fn = obj_source(dev)          # pass device
            else:
                obj_fn = obj_source() if callable(obj_source) else obj_source
        except (ValueError, TypeError):
            obj_fn = obj_source
    else:
        raise TypeError("obj_source must be callable")

    # silence worker stdout / stderr -----------------------------------------
    pid = str(os.getpid())
    sys.stdout = open(f"/tmp/cma_{pid}.out", "a")
    sys.stderr = open(f"/tmp/cma_{pid}.err", "a")

    # main loop ---------------------------------------------------------------
    while stop_q.empty():
        if param_q.empty():
            time.sleep(0.01)
            continue
        sid, x = param_q.get()
        fitness = obj_fn(x, dev)
        result_q.put((sid, fitness))


class ParallelEvaluator:
    """
    Farm-outs objective evaluations to N workers.

    obj_source:
        * plain callable  -> used directly in every worker
        * zero-arg factory -> called *inside each worker* to build
                              the real objective (useful for heavy setup)
    """

    def __init__(self, obj_source, n_workers: int = 8, show_progress: bool = True):
        self.param_q, self.result_q, self.stop_q = Queue(), Queue(), Queue()
        self.show_progress = show_progress
        self.workers = [Process(target=_worker, args=(self.param_q, self.result_q, self.stop_q, obj_source, wid)) for wid in range(n_workers)]
        for w in self.workers:
            w.start()

    def evaluate(self, params: Sequence[np.ndarray], repeats: int = 1) -> list[float]:
        """Return mean fitness for every element in *params* (lower = better)."""
        n_tasks = len(params) * repeats
        agg = [0.0] * len(params)
        for sid, p in enumerate(params):
            for _ in range(repeats):
                self.param_q.put((sid, p))
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
        """
        Try to shut workers down gracefully, then force-kill if they ignore us.

        Parameters
        ----------
        grace : float
            Seconds to wait for a clean exit before calling ``terminate()``.
        """
        # 1. signal stop (one token is enough—workers only test `.empty()`)
        self.stop_q.put(True)

        # 2. wait for graceful exit
        for w in self.workers:
            w.join(timeout=grace)

        # 3. hard kill any stragglers
        for w in self.workers:
            if w.is_alive():
                print(f"[ParallelEvaluator] Worker {w.pid} still alive → terminate()")
                w.terminate()
                w.join()

        # 4. tidy up log files (ignore if still held/open)
        for w in self.workers:
            for ext in (".out", ".err"):
                try:
                    os.remove(f"/tmp/cma_{w.pid}{ext}")
                except FileNotFoundError:
                    pass
                except PermissionError:
                    # File might still be locked on some OSes—skip
                    pass

        # 5. close queues and detach background threads
        for q in (self.param_q, self.result_q, self.stop_q):
            q.close()
            q.join_thread()          # prevents hanging on interpreter exit


def cma_es_optimize(obj_fn: Callable[[np.ndarray, torch.device], float],
                    x0: np.ndarray,
                    sigma0: float = 0.5,
                    lower_bounds: np.ndarray = None,
                    upper_bounds: np.ndarray = None,
                    popsize: int = 64,
                    max_epochs: int = 1000,
                    repeats: int = 4,
                    target: float = None,
                    n_workers: int = 8):
    """
    Parallel CMA-ES optimisation with progress-bar and clean shutdown.
    Returns (best_params, best_fitness).
    Solves minimisation problem.
    """
    evaluator = ParallelEvaluator(obj_fn, n_workers=n_workers)
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
        evaluator.close()          # ensures workers exit even on errors
    return best_params, best_fit


def rastrigin(x: np.ndarray, *_):
    """Shifted + negated so CMA-ES *minimises* (-reward == cost)."""
    z = x - 10.0
    return 10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))


class Toy:
    def __init__(self): self.offset = 3.0
    def obj(self, x, *_): return np.sum((x - self.offset)**2)


class Heavy:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = torch.randn(1000, 1000, device=device)

    def obj(self, x: np.ndarray, _device_from_pool):
        t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return float(((t @ t) + self.model[0, 0]).cpu())


def heavy_factory(device: torch.device):
    return Heavy(device).obj      # returns the bound method


if __name__ == "__main__":
    best_x, best_f = cma_es_optimize(
        rastrigin,
        # Toy().obj,
        # heavy_factory,
        x0=np.zeros(100),
        sigma0=0.5,
        popsize=101,
        max_epochs=1000,
        repeats=1,
        n_workers=4,          # adjust to GPUs/CPU cores you have
        target=1e-8           # optional early-stop
    )
    print("Best fitness:", best_f)
    print("Best params (first 6 dims):", best_x[:6])
