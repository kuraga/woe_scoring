# Main classes
from .model import Model
from .selector import Selector

# Functions previously in functions.py, now organized into modules
# These imports make the modules themselves available under the 'woe_scoring.core.model' namespace.
# For example, after 'from woe_scoring.core import model', you can use 'model.gini_calculator'.
from . import gini_calculator
from . import feature_analyzer
from . import model_analyzer
from . import sql_generator
from . import scorecard_builder
from . import excel_builder

# Top-level functions that remained in the original functions.py (or were modified)
from .functions import (
    save_scorecard, # Orchestrates scorecard_builder and excel_builder
    calc_model_results # Utility for Model class
)

# Define __all__ to control `from woe_scoring.core.model import *` behavior.
# This list explicitly states what symbols are exported when `import *` is used.
__all__ = [
    "Model",
    "Selector",
    "gini_calculator",
    "feature_analyzer",
    "model_analyzer",
    "sql_generator",
    "scorecard_builder",
    "excel_builder",
    "save_scorecard",
    "calc_model_results",
]
