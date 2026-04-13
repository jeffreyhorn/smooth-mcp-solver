__version__ = "0.1.0"

from smooth_mcp.diff import make_mcp_solver_diff
from smooth_mcp.smoothing import smooth_max, smooth_min, smooth_proj, smoothed_residual
from smooth_mcp.solver import MCPResult, solve_mcp

__all__ = [
    "MCPResult",
    "solve_mcp",
    "make_mcp_solver_diff",
    "smooth_max",
    "smooth_min",
    "smooth_proj",
    "smoothed_residual",
]
