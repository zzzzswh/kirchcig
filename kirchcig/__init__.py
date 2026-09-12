"""kirchcig: GPU Kirchhoff migration to common-image gathers, and its exact adjoint.

    from kirchcig import migrate, KirchhoffCIG

``migrate`` is the one-shot function; ``KirchhoffCIG`` is the operator pair for
inversion. The PyTorch wrapper lives in ``kirchcig.torch``.
"""
from ._version import __version__
from ._operator import KirchhoffCIG, aperture_mask, migrate
from ._traveltime import (analytic_traveltime, eikonal_traveltime,
                          emergence_angles, traveltime_tables)
from ._engine_numpy import NumpyEngine


def cuda_available() -> bool:
    """True when the ``cuda`` engine can be used (CuPy + a visible GPU)."""
    from ._engine_cuda import cuda_available as _avail
    return _avail()


__all__ = [
    "__version__",
    "KirchhoffCIG",
    "migrate",
    "aperture_mask",
    "cuda_available",
    "NumpyEngine",
    "traveltime_tables",
    "analytic_traveltime",
    "eikonal_traveltime",
    "emergence_angles",
]
