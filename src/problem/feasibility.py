"""
Feasibility evaluation for the X2 free-piston driver problem.

Provides a constraint vector g(x) and a binary feasibility check.  The
convention is the one used by Chocat et al. (2015) and Arnold & Hansen
(2012):

    g_j(x) <= 0   <=>   constraint j is satisfied at x.

This module is the single source of truth for "is this candidate worth
evaluating on SPARK / PITOT3?".  Infeasible candidates have no fitness
computed; their constraint violations feed the CHT covariance update.

Constraint layout
-----------------
The returned vector has length 6 + 2*n (with n = len(x) = 6):

    j=0     driver_p - reservoir_p              (driver pressure must be <= reservoir pressure)
    j=1     D_throat - 0.085                    (throat diameter upper bound, m)
    j=2     -D_throat                           (throat diameter must be > 0; safety net)
    j=3     5 - compression_ratio               (lower CR bound)
    j=4     compression_ratio - 70              (upper CR bound)
    j=5     p4 - bounds[2][1]                   (hard p4 ceiling)
    j=6..6+n-1     1 - x_i                      (normalised lower box bound for each var)
    j=6+n..6+2n-1  x_i - 2                      (normalised upper box bound for each var)

Order of operations
-------------------
Box bounds are checked first.  Violations of the [1, 2] box make
variable_untransformation() ill-defined (it has driver_p-coupled
divisions for p4 and reservoir_p), so the physical-space constraints
cannot be evaluated reliably.  When box bounds are violated we still
return a full-length vector but with the physical-space slots set to
+inf (clearly violated, but not used because the candidate is already
infeasible).
"""

import numpy as np

from problem.transforms import variable_untransformation


N_PHYSICAL_CONSTRAINTS = 6
COMPRESSION_GAMMA = 1.667        # heat capacity ratio used in compression-ratio approximation
COMPRESSION_LOWER, COMPRESSION_UPPER = 5.0, 70.0
D_THROAT_UPPER = 0.085


def evaluate_constraints(x_normalised, bounds):
    """Return the signed constraint vector for candidate x.

    Parameters
    ----------
    x_normalised : sequence of float
        Candidate in normalised [1, 2]^n space.
    bounds : list of (lo, hi)
        Physical-space bounds (used for the p4 ceiling and
        un-transformation).

    Returns
    -------
    g : np.ndarray, shape (6 + 2*n,)
        Signed constraint values.  g[j] <= 0  <=>  constraint j satisfied.
    """
    x = np.asarray(x_normalised, dtype=float)
    n = len(x)
    g = np.empty(N_PHYSICAL_CONSTRAINTS + 2 * n, dtype=float)

    # --- Box bounds (always evaluable) -------------------------------------
    # Lower bound 1 - x_i  <= 0   <=>   x_i >= 1
    # Upper bound x_i - 2  <= 0   <=>   x_i <= 2
    g[N_PHYSICAL_CONSTRAINTS:N_PHYSICAL_CONSTRAINTS + n]     = 1.0 - x
    g[N_PHYSICAL_CONSTRAINTS + n:N_PHYSICAL_CONSTRAINTS + 2*n] = x - 2.0

    # --- Physical-space constraints ---------------------------------------
    # Only meaningful when x is inside the normalised box, because the
    # un-transformation couples p4 and reservoir_p to driver_p.
    box_ok = np.all(g[N_PHYSICAL_CONSTRAINTS:] <= 0.0)
    if not box_ok:
        # Mark physical constraints as "violated" without computing them.
        # Using +inf is intentional: it is unambiguously > 0 and survives
        # any later sign-aware rank ordering, but never gets aggregated
        # because is_feasible() will already return False from the box.
        g[:N_PHYSICAL_CONSTRAINTS] = np.inf
        return g

    # x is inside the box, so un-transformation is well-defined.
    x_phys = variable_untransformation(x, bounds)
    percent_he, driver_p, p4, d_throat, reservoir_p, _buffer_length = x_phys

    g[0] = driver_p - reservoir_p
    g[1] = d_throat - D_THROAT_UPPER
    # NOTE: With box bounds satisfied, x[3] in [1, 2] => d_throat in [0.05, 0.085]
    # so this is structurally redundant, but kept as a safety net in case
    # the bounds[3] mapping ever changes.
    g[2] = -d_throat

    pressure_ratio = p4 / driver_p
    compression_ratio = pressure_ratio ** (1.0 / COMPRESSION_GAMMA)
    g[3] = COMPRESSION_LOWER - compression_ratio
    g[4] = compression_ratio - COMPRESSION_UPPER

    g[5] = p4 - bounds[2][1]

    return g


def is_feasible(g, tol=0.0):
    """Return True iff every constraint is satisfied within tolerance.

    A candidate is feasible iff g_j <= tol for all j.  The default tol=0
    matches Chocat's strict definition.
    """
    return bool(np.all(np.asarray(g) <= tol))
