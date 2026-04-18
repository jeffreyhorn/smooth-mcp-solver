__version__ = "0.1.0"

from smooth_mcp._kernel import preflight_validate
from smooth_mcp._types import MCPResult, SolveInfo
from smooth_mcp.diff import make_mcp_solver_diff
from smooth_mcp.forward import make_mcp_solver
from smooth_mcp.smoothing import smooth_max, smooth_min, smooth_proj, smoothed_residual
from smooth_mcp.solver import solve_mcp

__all__ = [
    "MCPResult",
    "SolveInfo",
    "solve_mcp",
    "make_mcp_solver",
    "make_mcp_solver_diff",
    "preflight_validate",
    "smooth_max",
    "smooth_min",
    "smooth_proj",
    "smoothed_residual",
]
