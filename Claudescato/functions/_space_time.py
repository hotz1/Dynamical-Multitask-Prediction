"""Helpers for building space-time input sets."""

import numpy as np


def space_locs_2d(space_locs):
    """Return spatial locations as an (N, d) float64 array."""
    space_locs = np.asarray(space_locs, dtype=np.float64)
    return space_locs[:, None] if space_locs.ndim == 1 else space_locs


def space_time_inputs(space_locs, time_instant):
    """Return the (N, 1 + d) inputs [t, s] for all locations at one time."""
    space_locs = space_locs_2d(space_locs)
    t_col = np.full((space_locs.shape[0], 1), float(time_instant))
    return np.hstack([t_col, space_locs])


def space_time_grid(space_locs, time_instants):
    """Return inputs for every (location, time) pair, space varying fastest.

    This matches MATLAB's column-major vectorization Y(:) of a
    (space x time) matrix.
    """
    return np.vstack([space_time_inputs(space_locs, t) for t in time_instants])
