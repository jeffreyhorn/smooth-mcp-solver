__version__ = "0.1.0"

from smooth_mcp.core import (
    MCPResult,
    make_mcp_solver_diff,
    smooth_max,
    smooth_min,
    smooth_proj,
    smoothed_residual,
    solve_mcp,
    solve_mcp_diff,
)

__all__ = [
    "MCPResult",
    "solve_mcp",
    "make_mcp_solver_diff",
    "solve_mcp_diff",
    "smooth_max",
    "smooth_min",
    "smooth_proj",
    "smoothed_residual",
]
