from typing import Optional, Tuple, Callable, Any

import asyncio
import concurrent.futures
import torch


def resolve_devices_and_workers(device: torch.device, batch_size_per_gpu: int) -> Tuple[Optional[torch.device], int]:
    """Return (devices_param, num_workers) for AsyncMultiWrapper using per-GPU worker count.

    - device == cuda with no index → round-robin across all GPUs, num_workers = num_gpus * batch_size_per_gpu
    - device == cuda:X → pin to that GPU, num_workers = batch_size_per_gpu
    - device == cpu → single worker
    """
    if device.type == "cuda":
        if device.index is None:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            num_workers = max(1, (n_gpus or 1) * max(1, batch_size_per_gpu))
            return None, num_workers
        else:
            num_workers = max(1, batch_size_per_gpu)
            return torch.device(f"cuda:{device.index}"), num_workers
    return torch.device("cpu"), 1


def run_async_in_any_context(coro_fn: Callable[[], Any]) -> Any:
    """Run an async coroutine function regardless of existing event loop.

    Expects a no-arg function that returns an awaitable when called.
    """
    try:
        asyncio.get_running_loop()

        def _thread_run():
            return asyncio.run(coro_fn())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_thread_run)
            return fut.result()
    except RuntimeError:
        return asyncio.run(coro_fn())
