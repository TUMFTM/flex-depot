from importlib.metadata import PackageNotFoundError, version

from flex_dep_opt.batch import run_batch
from flex_dep_opt.config.settings import Settings
from flex_dep_opt.errors import SimulationError
from flex_dep_opt.io.inputs import SimulationInputs, load_inputs
from flex_dep_opt.io.results import SimulationResult
from flex_dep_opt.workflows.mpc_workflow import simulate
from flex_dep_opt.workflows.postprocessing_workflow import Evaluation, evaluate

try:
    __version__ = version("flex-depot")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Evaluation",
    "Settings",
    "SimulationError",
    "SimulationInputs",
    "SimulationResult",
    "evaluate",
    "load_inputs",
    "run_batch",
    "simulate",
]
