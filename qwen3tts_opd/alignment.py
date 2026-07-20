from __future__ import annotations


def frame_prediction_slice(prefill_length: int, num_frames: int) -> slice:
    """Select states immediately before each replayed codec frame."""
    if prefill_length < 1:
        raise ValueError("prefill_length must be positive")
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    return slice(prefill_length - 1, prefill_length + num_frames - 1)
