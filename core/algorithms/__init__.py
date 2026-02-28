from core.algorithms.base import Algorithm


def build_algorithm(*args, **kwargs):
    from core.algorithms.factory import build_algorithm as _build_algorithm

    return _build_algorithm(*args, **kwargs)

__all__ = ["Algorithm", "build_algorithm"]
