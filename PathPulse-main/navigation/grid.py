"""
PathPulse — Hospital Grid Definition
======================================
Defines the 2D hospital floor-plan as a NumPy array, marks obstacle
cells (walls / restricted zones), and exports a dictionary mapping
department names to their ``(row, col)`` coordinates on the grid.

Grid Convention
---------------
* ``0`` — Walkable corridor / room entrance
* ``1`` — Wall or restricted zone (impassable)

The grid is **20 × 20** (rows 0-19, cols 0-19).  Row 0 is the top of
the map, column 0 is the left edge.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Grid dimensions
# ---------------------------------------------------------------------------

GRID_ROWS: int = 20
GRID_COLS: int = 20

# ---------------------------------------------------------------------------
# Hospital floor-plan (20 × 20)
# ---------------------------------------------------------------------------
# Layout concept:
#   - Top-left area      :  Dispensary (start) + Pharmacy
#   - Top-right area     :  Laboratory + Radiology
#   - Centre             :  Main corridors (open space)
#   - Left-centre        :  Emergency Ward
#   - Right-centre       :  Operation Theater
#   - Bottom-left        :  Maternity Ward
#   - Bottom-right       :  ICU
#   - Bottom-centre      :  Cardiology
#
# Walls are placed to create realistic corridors and room separations.
# The grid is hand-designed so that every room is reachable from the
# Dispensary, but the paths are non-trivial.

HOSPITAL_GRID: np.ndarray = np.array([
    # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
    [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # row 0
    [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # row 1
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 2
    [ 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],  # row 3
    [ 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],  # row 4
    [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # row 5
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 6
    [ 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],  # row 7
    [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # row 8
    [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # row 9
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 10
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 11
    [ 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],  # row 12
    [ 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # row 13
    [ 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # row 14
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 15
    [ 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],  # row 16
    [ 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],  # row 17
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 18
    [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # row 19
], dtype=np.int8)


# ---------------------------------------------------------------------------
# Room / department coordinates  →  (row, col)
# ---------------------------------------------------------------------------
# All coordinates point to walkable cells (value 0 in the grid above).

ROOMS: Dict[str, Tuple[int, int]] = {
    "Dispensary":        (0, 1),     # Top-left — fixed origin for navigation
    "Pharmacy":          (1, 2),     # Near dispensary
    "Laboratory":        (0, 17),    # Top-right wing
    "Radiology":         (1, 17),    # Adjacent to laboratory
    "Emergency Ward":    (8, 3),     # Left-centre block
    "Operation Theater": (8, 16),    # Right-centre block
    "Maternity Ward":    (13, 3),    # Bottom-left block
    "Cardiology":        (14, 9),    # Bottom-centre
    "ICU":               (13, 16),   # Bottom-right block
    "General Ward":      (18, 9),    # Bottom corridor
    "Reception":         (6, 9),     # Central corridor hub
}

# The dispensary is always the starting point for supply-cart navigation.
START_ROOM: str = "Dispensary"
START_COORDS: Tuple[int, int] = ROOMS[START_ROOM]


# ---------------------------------------------------------------------------
# Room colours for visualisation (Matplotlib RGBA-compatible hex codes)
# ---------------------------------------------------------------------------
# Each room gets a distinct colour so the grid plot is easy to read.

ROOM_COLORS: Dict[str, str] = {
    "Dispensary":        "#ff2a2a",   # Neon crimson (start)
    "Pharmacy":          "#888888",   # Gray
    "Laboratory":        "#666666",   # Darker Gray
    "Radiology":         "#999999",   # Medium Gray
    "Emergency Ward":    "#ff5252",   # Interactive Red
    "Operation Theater": "#ff8f8f",   # Soft Coral
    "Maternity Ward":    "#777777",   # Gray
    "Cardiology":        "#d1d5db",   # Light Gray
    "ICU":               "#555555",   # Dark Gray
    "General Ward":      "#444444",   # Deep Gray
    "Reception":         "#aaaaaa",   # Light Gray
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_room_names() -> List[str]:
    """Return a sorted list of all room / department names.

    Returns
    -------
    list[str]
        Alphabetically sorted room names.
    """
    return sorted(ROOMS.keys())


def get_destination_rooms() -> List[str]:
    """Return room names available as navigation destinations.

    The dispensary (origin) is excluded since you can't navigate *to*
    the starting location.

    Returns
    -------
    list[str]
        Sorted list of destination room names.
    """
    return sorted(name for name in ROOMS if name != START_ROOM)


def is_valid_cell(row: int, col: int) -> bool:
    """Check whether a cell is within grid bounds and walkable.

    Parameters
    ----------
    row : int
        Row index (0-based).
    col : int
        Column index (0-based).

    Returns
    -------
    bool
        ``True`` if the cell is in bounds and not a wall.
    """
    if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
        return HOSPITAL_GRID[row, col] == 0
    return False


def get_neighbors(row: int, col: int) -> List[Tuple[int, int]]:
    """Return walkable 4-connected neighbors of a cell.

    Movement is restricted to the four cardinal directions (no
    diagonals), which aligns with hospital corridor movement.

    Parameters
    ----------
    row : int
        Current row index.
    col : int
        Current column index.

    Returns
    -------
    list[tuple[int, int]]
        List of ``(row, col)`` tuples for valid adjacent cells.
    """
    neighbors: List[Tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if is_valid_cell(nr, nc):
            neighbors.append((nr, nc))
    return neighbors
