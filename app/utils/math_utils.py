from __future__ import annotations

import math


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
