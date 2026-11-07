__all__ = ["DifferentiableGPRFDTD", "FDTDConfig", "format_bscan"]

def __getattr__(name):
    if name in __all__:
        from . import operator as _operator
        return getattr(_operator, name)
    raise AttributeError(name)
