__version__ = "0.1.6"
__author__ = 'Forest Agostinelli'

# run registers
from . import heuristics  # noqa: F401

from deepxube.factories.nnet_input_factory import register_nnet_input_dynamic
from deepxube.factories.factory_utils import import_local_modules
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import_local_modules("domains/")
import_local_modules("heuristics/")
register_nnet_input_dynamic()
