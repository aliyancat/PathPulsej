"""
PathPulse — A* Search
======================
Implements the A* search algorithm on the hospital grid.

Algorithm overview
------------------
A* is an informed search that combines the actual cost from start
``g(n)`` with a heuristic estimate to the goal ``h(n)``.  It expands
the node with the lowest ``f(n) = g(n) + h(n)``.  With an admissible
and consistent heuristic (Manhattan distance on a grid), A* is
guaranteed to find the shortest path.

Heuristic
---------
Manhattan distance:  ``|row1 - row2| + |col1 - col2|``

Signature (per architecture.md)
-------------------------------
    search(grid, start, goal)  →  (path, explored_count)
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from navigation.grid import get_neighbors


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def manhattan_distance(
    a: Tuple[int, int],
    b: Tuple[int, int],
) -> int:
    """Compute the Manhattan distance between two grid cells.

    Parameters
    ----------
    a : tuple[int, int]
        ``(row, col)`` of the first cell.
    b : tuple[int, int]
        ``(row, col)`` of the second cell.

    Returns
    -------
    int
        ``|a.row - b.row| + |a.col - b.col|``
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Tuple[Optional[List[Tuple[int, int]]], int, List[Tuple[int, int]]]:
    """Run A* Search from *start* to *goal*.

    Parameters
    ----------
    grid : np.ndarray
        2D hospital grid (0 = walkable, 1 = wall).
    start : tuple[int, int]
        ``(row, col)`` of the origin cell (Dispensary).
    goal : tuple[int, int]
        ``(row, col)`` of the destination cell.

    Returns
    -------
    path : list[tuple[int, int]] or None
        Ordered list of ``(row, col)`` coordinates from *start* to
        *goal* (inclusive at both ends).  ``None`` if no path exists.
    explored_count : int
        Total number of distinct nodes that were expanded (popped
        from the frontier).
    explored_order : list[tuple[int, int]]
        Cells in the order they were explored — used for the animated
        visualisation of the search frontier.
    """
    # Priority queue entries: (f_value, tie_breaker, (row, col))
    counter: int = 0
    frontier: List[Tuple[int, int, Tuple[int, int]]] = []
    heapq.heappush(
        frontier,
        (manhattan_distance(start, goal), counter, start),
    )
    counter += 1

    # g_score tracks the shortest known distance from start to each node
    g_score: Dict[Tuple[int, int], int] = {start: 0}

    # came_from tracks the parent of each visited cell for path reconstruction
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    # Track exploration order for visualisation
    explored_order: List[Tuple[int, int]] = []

    # Closed set — nodes that have already been fully expanded
    closed: set[Tuple[int, int]] = set()

    while frontier:
        _f, _tie, current = heapq.heappop(frontier)

        # Skip if we already expanded this node (duplicate in the heap)
        if current in closed:
            continue

        closed.add(current)
        explored_order.append(current)

        # Goal check
        if current == goal:
            path: List[Tuple[int, int]] = _reconstruct_path(came_from, current)
            return path, len(explored_order), explored_order

        # Expand neighbors
        current_g: int = g_score[current]
        for neighbor in get_neighbors(current[0], current[1]):
            tentative_g: int = current_g + 1  # uniform step cost = 1

            if tentative_g < g_score.get(neighbor, float("inf")):  # type: ignore[arg-type]
                # Found a better path to this neighbor
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f: int = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(frontier, (f, counter, neighbor))
                counter += 1

    # No path found — goal is unreachable
    return None, len(explored_order), explored_order


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_path(
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    current: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Trace back from *current* to the start using the ``came_from`` map.

    Parameters
    ----------
    came_from : dict
        Mapping of child → parent produced during search.
    current : tuple[int, int]
        The goal cell (end of the path).

    Returns
    -------
    list[tuple[int, int]]
        Full path from start to goal, inclusive.
    """
    path: List[Tuple[int, int]] = [current]
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path
