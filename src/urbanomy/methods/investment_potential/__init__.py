# investment_potential/__init__.py
from importlib.metadata import PackageNotFoundError, version as _v
try:
    __version__ = _v("investment_potential")
except PackageNotFoundError:  # fallback used during local development
    __version__ = "0.0.1"

from .land_use_score import LandUseScoreAnalyzer
from .investment_metrics import (
    InvestmentAttractivenessAnalyzer,
    calculate_investment_metrics,
)
from .constants import DEFAULT_BENCHMARKS_RU, LAND_USE_TO_POTENTIAL_COLUMN
from urbanomy.utils.investment_input import prepare_investment_input
