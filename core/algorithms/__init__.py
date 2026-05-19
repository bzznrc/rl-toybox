def build_algorithm(*args, **kwargs):
    from core.algorithms.factory import build_algorithm as _build_algorithm

    return _build_algorithm(*args, **kwargs)


def __getattr__(name: str):
    if name == "Algorithm":
        from core.algorithms.base import Algorithm

        return Algorithm
    raise AttributeError(name)


__all__ = ["Algorithm", "build_algorithm"]
