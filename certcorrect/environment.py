"""Run metadata written into every result file: machine, library versions (fea_engine and rom_engine are cited in the
paper), BLAS threads and the sparse direct solver."""

import importlib.util
import os
import platform
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def single_thread() -> None:
    """Pin BLAS/OpenMP to one thread; call before NumPy is imported (timing experiments X6)."""
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"


def _blas_info() -> dict | None:
    import numpy

    try:
        blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]
    except (TypeError, KeyError):              # NumPy < 1.26 has no dict output
        return None
    return {"name": blas.get("name"), "version": blas.get("version")}


def environment_info() -> dict:
    """Plain-data description of the machine and the software stack."""
    import numpy
    import scipy

    import fea_engine
    import rom_engine

    import certcorrect

    return {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "fea_engine": getattr(fea_engine, "__version__", None),
            "rom_engine": getattr(rom_engine, "__version__", None), "certcorrect": certcorrect.__version__,
            "blas": _blas_info(), "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "sparse_direct_solver": "UMFPACK" if importlib.util.find_spec("scikits") else "SuperLU (scipy.sparse.linalg)"}
