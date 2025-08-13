
$init = @'
try:
    from ._version import version as __version__  # written by setuptools_scm (optional)
except Exception:
    __version__ = "0.0.0"

# Export primary classes; update import paths if filenames differ.
from .processing.RF_processing import RFProcessor
from .processing.NN_processing import NNProcessor

__all__ = ["RFProcessor", "NNProcessor", "__version__"]
'@

$init | Set-Content -Path .\src\uav_biophysical_estimation\__init__.py -Encoding UTF8