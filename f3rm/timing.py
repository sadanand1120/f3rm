from collections import deque

from nerfstudio.utils import writer


_ORIGINAL_PUT_TIME = getattr(writer, "_f3rm_original_put_time", writer.put_time)
writer._f3rm_original_put_time = _ORIGINAL_PUT_TIME
_TIMING_BUFFERS: dict[str, deque[float]] = {}
_WRITER_TIME_BUFFERS: dict[str, deque[float]] = {}
_TIMING_TOTALS: dict[str, float] = {}
_TIMING_COUNTS: dict[str, int] = {}


def _get_buffer(buffers: dict[str, deque[float]], name: str) -> deque[float]:
    max_buffer_size = max(1, int(writer.GLOBAL_BUFFER.get("max_buffer_size", 1)))
    buffer = buffers.get(name)
    if buffer is None or buffer.maxlen != max_buffer_size:
        buffer = deque(buffer or (), maxlen=max_buffer_size)
        buffers[name] = buffer
    return buffer


def _should_write(step: int) -> bool:
    if step == 0:
        return True
    steps_per_log = writer.GLOBAL_BUFFER.get("steps_per_log")
    if steps_per_log is not None and int(steps_per_log) > 0 and step % int(steps_per_log) == 0:
        return True
    max_iter = writer.GLOBAL_BUFFER.get("max_iter")
    return max_iter is not None and step == int(max_iter) - 1


def _put_average_scalar(name: str, value: float, step: int, buffers: dict[str, deque[float]]) -> float:
    buffer = _get_buffer(buffers, name)
    buffer.append(value)
    avg = sum(buffer) / len(buffer)
    if _should_write(step):
        writer.put_scalar(name=name, scalar=avg, step=step)
    return avg


def _throttled_writer_put_time(name: str, duration: float, step: int, avg_over_steps: bool = True, update_eta: bool = False):
    if not avg_over_steps or "events" not in writer.GLOBAL_BUFFER:
        _ORIGINAL_PUT_TIME(
            name=name,
            duration=duration,
            step=step,
            avg_over_steps=avg_over_steps,
            update_eta=update_eta,
        )
        return

    if isinstance(name, writer.EventName):
        name = name.value

    writer.GLOBAL_BUFFER["step"] = step
    avg = _put_average_scalar(name=name, value=duration, step=step, buffers=_WRITER_TIME_BUFFERS)
    writer.GLOBAL_BUFFER["events"][name] = {"buffer": list(_WRITER_TIME_BUFFERS[name]), "avg": avg}

    if update_eta:
        remain_iter = writer.GLOBAL_BUFFER["max_iter"] - step
        remain_time = remain_iter * avg
        if _should_write(step):
            writer.put_scalar(name=writer.EventName.ETA, scalar=remain_time, step=step)
        writer.GLOBAL_BUFFER["events"][writer.EventName.ETA.value] = writer._format_time(remain_time)


writer.put_time = _throttled_writer_put_time


def put_timing(name: str, duration: float, step: int, avg_over_steps: bool = True) -> None:
    """Log custom timing scalars without writing every per-step sample to event storage."""
    _TIMING_TOTALS[name] = _TIMING_TOTALS.get(name, 0.0) + duration
    _TIMING_COUNTS[name] = _TIMING_COUNTS.get(name, 0) + 1
    if not avg_over_steps:
        writer.put_time(name=name, duration=duration, step=step, avg_over_steps=False)
        return

    _put_average_scalar(name=name, value=duration, step=step, buffers=_TIMING_BUFFERS)


def flush_final_timings(step: int) -> None:
    """Write full-run timing aggregates once near the end of training."""
    for name in sorted(_TIMING_TOTALS):
        total = _TIMING_TOTALS[name]
        count = _TIMING_COUNTS[name]
        if count <= 0:
            continue
        base_name = name.removeprefix("Timing/")
        writer.put_scalar(name=f"Final Timing/{base_name}/avg_s", scalar=total / count, step=step)
        writer.put_scalar(name=f"Final Timing/{base_name}/total_s", scalar=total, step=step)
        writer.put_scalar(name=f"Final Timing/{base_name}/count", scalar=count, step=step)

    _TIMING_TOTALS.clear()
    _TIMING_COUNTS.clear()
