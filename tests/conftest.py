"""Put ``src/`` on sys.path for the test session.

The package is not installed and ``src`` is not a package, so the modules
under test are imported as top-level names (``main``, ``algorithm.cmaes``,
``problem.l1d_job``, ...). Doing it here once keeps it out of every test
module.

Note: this must not be done by exporting PYTHONPATH=src. The gdtk
environment is configured on PATH, and prepending src/ to PYTHONPATH
shadows it, which makes every l1d4-prep subprocess fail on import.
"""
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
