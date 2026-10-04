"""
Volume-conserving geometry construction for the X2 driver in L1d4.

Implements the methodology from Hodson 2025 §4.3.1: every discontinuous
area change in the simplified-cylindrical X2 driver model is replaced by
a gradual linear ramp whose endpoints are placed to preserve the internal
volume of the cylinder either side.  The single helper
:func:`volume_conserving_ramp` is the executable form of Eq 4.12-4.15.

:func:`build_break_points` walks the X2 transition table once per
individual and emits the (x, D) pairs that the L1d job script turns into
``add_break_point`` calls.  ``buffer_length`` and ``D_throat`` are the
only design variables that affect geometry; everything else is a facility
constant.

Reference data (Table 4.5, no buffer studs, no orifice plate) is
reproduced by ``build_break_points(buffer_length=0.0, D_throat=None)``;
this is asserted by the smoke test at the bottom of the file.
"""
import numpy as np

# Facility constants (Hodson Table 4.4 / 4.5)

# Reservoir-side area transitions: (x_nominal, R_upstream, R_downstream).
# All are reductions (downstream is smaller), all use V_buffer = 0.
RESERVOIR_GRADIENTS = [
    (-5.8007, 0.158,   0.122),
    (-5.1817, 0.122,   0.090),
    (-5.1407, 0.090,   0.07805),
]

# Buffer-plate gradient: the only transition affected by V_buffer.
BUFFER_PLATE_X_NOMINAL = -0.1120
R_COMPRESSION = 0.2568 / 2   # = 0.1284
R_SHOCK       = 0.0850 / 2   # = 0.0425

# Fixed anchors with no gradient applied.  The launcher exit anchors
# (-4.8332, D=0.1561) and (-4.808, D=0.2568) bracket a pre-baked 25.2 mm
# ramp from Eq 4.9 that L1d realises automatically via its piecewise-
# linear interpretation of consecutive break-points.
RESERVOIR_START_ANCHOR = (-8.7188, 0.316)
LAUNCHER_SMALL_D       = (-4.8332, 0.1561)
LAUNCHER_LARGE_D       = (-4.808,  0.2568)   # = piston rear initial position
PISTON_FRONT_X         = -4.587
PD_X                   = 0.0

# Buffer-stud parameters (X2 standard six-stud configuration).
N_BUFFER_STUDS = 6
D_BUFFER_STUD  = 0.050

# Hodson methodology constant.
M_ASPECT = 2.0

# Orifice-plate parameters.  Calibrated against two Hodson reference
# files (x2_condition1.py with buffer = 100 mm, at D_throat = 65 mm and
# D_throat = 77 mm).  The plate's geometry is ASYMMETRIC across the
# upstream and downstream sides:
#
#   * Upstream side: volume-conserving ramp (Eq 4.12-4.15) with a fixed
#     sharp-transition position X_ORIFICE_UPSTREAM_NOMINAL and fixed
#     chamfer slope 1/M_ORIFICE.  As D_throat shrinks, the chamfer gets
#     longer (more radial distance to cover at slope 1) and both ramp
#     endpoints move outward, but the implicit sharp-transition stays
#     put.  Verified across both reference D_throat values.
#
#   * Downstream side: fixed throat-side and outer endpoints.  The
#     plate's downstream face is hardware-bolted to the shock tube at
#     X_ORIFICE_DOWNSTREAM_OUTER, and the constant-D bore always ends
#     at X_ORIFICE_DOWNSTREAM_BORE_END (100 mm of bore measured from
#     the downstream face).  The chamfer slope adjusts with D_throat
#     to span the variable radial step between these two fixed points.
#
# This asymmetric model reproduces both reference files' four orifice
# break-points exactly (up to 16 microns, which is below mesh
# resolution and reflects rounding in the reference).
X_ORIFICE_UPSTREAM_NOMINAL    = -0.010    # x_nominal of upstream volume-conserving ramp
X_ORIFICE_DOWNSTREAM_BORE_END = +0.100    # x_r_T7: small-D end of downstream ramp
X_ORIFICE_DOWNSTREAM_OUTER    = +0.110    # x_R_T7: large-D end of downstream ramp

# Slope (radius:length) of the upstream volume-conserving chamfer.
# m = 1 matches both reference files.
M_ORIFICE = 1.0

# Bulk x-shift applied to the entire orifice plate (upstream nominal,
# downstream bore end, and downstream outer all translate together as
# a rigid body).  Negative shifts the plate UPSTREAM relative to the
# primary diaphragm at x = 0, keeping PD comfortably inside the
# constant-bore throat across the full D_throat optimisation range.
#
# At ORIFICE_X_SHIFT = 0 the geometry reproduces the Hodson reference
# files exactly.  Non-zero shifts are a deliberate divergence and the
# smoke-test reference assertions are skipped accordingly.
#
# Tuned for the optimisation's D_throat lower bound of 0.050 m: at
# that lower bound the upstream throat-side endpoint x_r_T6 sits
# ~10 mm upstream of PD, so the diaphragm rupture happens cleanly
# inside the constant-D throat instead of inside the variable-area
# upstream chamfer.
ORIFICE_X_SHIFT = -0.010

# Minimal-model shock-tube extent past the primary diaphragm.
L_SHOCK_TUBE_END = 5.0


# Volume-conserving gradient (Hodson Eq 4.12 - 4.15)

def volume_conserving_ramp(R, r, x_nominal, direction, m=M_ASPECT, V_buffer=0.0):
    """One area transition.

    Parameters
    ----------
    R, r : float
        Radii either side of the transition.  ``R > r`` must hold.
    x_nominal : float
        Sharp-discontinuity x-coordinate from the cylindrical (Table 4.4)
        layout.
    direction : int
        ``+1`` for an expansion in the downstream direction (small-D
        upstream, large-D downstream); ``-1`` for a reduction.
    m : float
        Aspect ratio of the gradient (radius:length).  Hodson uses 2.
    V_buffer : float
        Volume to subtract from the upstream cylinder to account for
        buffer studs.  Zero for every transition except the buffer-plate.

    Returns
    -------
    (x_R, x_r) : tuple of float
        x-coordinates of the large-D and small-D endpoints of the ramp.
        For ``direction == -1`` ``x_R < x_r`` (large-D upstream); for
        ``direction == +1`` ``x_r < x_R`` (small-D upstream).
    """
    if R <= r:
        raise ValueError(f"volume_conserving_ramp expects R > r; got R={R}, r={r}")
    m_signed = m * direction
    a = (2 * (R**3 - r**3 + 3 * m_signed * V_buffer / (2 * np.pi))
         ) / (3 * (R**2 - r**2))
    x_R = (R - a) / m_signed + x_nominal
    x_r = (r - a) / m_signed + x_nominal
    return x_R, x_r


def buffer_stud_volume(buffer_length):
    """Total displaced volume of the six X2 buffer studs (Eq 4.13).

    ``buffer_length`` <= 0 disables the studs (V_buffer = 0).
    """
    if buffer_length <= 0:
        return 0.0
    return N_BUFFER_STUDS * 0.25 * np.pi * buffer_length * D_BUFFER_STUD**2


# Per-individual break-point assembly

def build_break_points(buffer_length, D_throat=None):
    """Return ``(break_points, derived)`` for one individual.

    Parameters
    ----------
    buffer_length : float
        Buffer-stud length in metres.  Zero or negative disables the
        studs (V_buffer = 0); otherwise V_buffer feeds Eq 4.12 for the
        buffer-plate transition only.
    D_throat : float or None
        Orifice-plate bore diameter in metres.  ``None`` means no orifice
        plate (the basic Hodson Table 4.5 configuration).

    Returns
    -------
    break_points : list of (x, D) tuples
        In ascending-x order, ready to be emitted as L1d
        ``add_break_point`` calls.
    derived : dict
        Computed positions the job-script template and the parser need
        (piston anchors, primary-diaphragm x, buffer-plate ramp endpoints,
        orifice centre when present).
    """
    V_b = buffer_stud_volume(buffer_length)
    bps = [RESERVOIR_START_ANCHOR]

    # Three reservoir-side gradients (all reductions, all V_buffer = 0).
    for x_nom, R_up, R_dn in RESERVOIR_GRADIENTS:
        x_R, x_r = volume_conserving_ramp(R_up, R_dn, x_nom, direction=-1)
        bps.append((x_R, 2 * R_up))
        bps.append((x_r, 2 * R_dn))

    # Launcher anchors - diameter ramp between them is interpolated by L1d.
    bps.append(LAUNCHER_SMALL_D)
    bps.append(LAUNCHER_LARGE_D)
    bps.append((PISTON_FRONT_X, 2 * R_COMPRESSION))

    # Buffer-plate gradient - the only stud-affected transition.
    x_R_T5, x_r_T5 = volume_conserving_ramp(
        R_COMPRESSION, R_SHOCK, BUFFER_PLATE_X_NOMINAL,
        direction=-1, V_buffer=V_b,
    )
    bps.append((x_R_T5, 2 * R_COMPRESSION))
    bps.append((x_r_T5, 2 * R_SHOCK))

    derived = {
        "piston_xL0":      LAUNCHER_LARGE_D[0],
        "piston_xR0":      PISTON_FRONT_X,
        "pd_x":            PD_X,
        "x_outer_buffer":  x_R_T5,
        "x_inner_buffer":  x_r_T5,
    }

    # Orifice plate, if a D_throat is supplied and is strictly smaller
    # than the shock-tube diameter.  At D_throat == 2*R_SHOCK the
    # orifice is degenerate (no area change), so we skip the ramp calls.
    #
    # Asymmetric geometry calibrated against two Hodson reference files:
    #
    #   * Upstream side: volume-conserving ramp with x_nominal fixed at
    #     X_ORIFICE_UPSTREAM_NOMINAL and slope 1/M_ORIFICE.  Both ramp
    #     endpoints (x_R_T6, x_r_T6) move outward as D_throat shrinks.
    #
    #   * Downstream side: fixed throat-side endpoint at
    #     X_ORIFICE_DOWNSTREAM_BORE_END and fixed outer endpoint at
    #     X_ORIFICE_DOWNSTREAM_OUTER.  The chamfer slope varies with
    #     D_throat (flatter at smaller D_throat).
    #
    # See the constant block above for derivation.  This model
    # reproduces both reference files' break-points exactly.
    if D_throat is not None and D_throat < 2 * R_SHOCK:
        R_throat = D_throat / 2.0

        # Upstream ramp: volume-conserving with direction=-1.
        # Returns (x_R, x_r) with x_R upstream (D=2*R_SHOCK side) and
        # x_r downstream (D=D_throat side).  ORIFICE_X_SHIFT translates
        # the implicit sharp-transition position upstream by a fixed
        # amount; both ramp endpoints shift with it.
        x_R_T6, x_r_T6 = volume_conserving_ramp(
            R_SHOCK, R_throat, X_ORIFICE_UPSTREAM_NOMINAL + ORIFICE_X_SHIFT,
            direction=-1, m=M_ORIFICE,
        )

        # Downstream ramp: hard-coded hardware endpoints, also shifted
        # by ORIFICE_X_SHIFT so the entire plate translates as a rigid
        # body (slope, bore length, and chamfer footprint unchanged).
        x_r_T7 = X_ORIFICE_DOWNSTREAM_BORE_END + ORIFICE_X_SHIFT
        x_R_T7 = X_ORIFICE_DOWNSTREAM_OUTER    + ORIFICE_X_SHIFT

        # Feasibility -- upstream chamfer must not encroach on the
        # buffer-plate small-D end; the two ramps must not eat the
        # entire throat span; downstream chamfer must end before the
        # shock-tube exit.
        if x_R_T6 <= x_r_T5:
            raise ValueError(
                f"Orifice upstream chamfer infeasible at "
                f"D_throat={D_throat:.4f} m: x_R_T6={x_R_T6:.6f} "
                f"<= x_inner_buffer={x_r_T5:.6f}."
            )
        if x_r_T6 >= x_r_T7:
            raise ValueError(
                f"Orifice throat degenerate at D_throat={D_throat:.4f} m: "
                f"upstream chamfer ends at x_r_T6={x_r_T6:.6f} which is "
                f">= downstream bore end x_r_T7={x_r_T7:.6f}."
            )
        if x_R_T7 >= L_SHOCK_TUBE_END:
            raise ValueError(
                f"Orifice downstream chamfer infeasible: "
                f"x_R_T7={x_R_T7:.6f} >= L_SHOCK_TUBE_END={L_SHOCK_TUBE_END:.6f}."
            )

        bps.append((x_R_T6, 2 * R_SHOCK))
        bps.append((x_r_T6, D_throat))
        bps.append((x_r_T7, D_throat))
        bps.append((x_R_T7, 2 * R_SHOCK))
        derived["x_orifice_centre"] = 0.5 * (x_r_T6 + x_r_T7)

    # Without an orifice we add an explicit anchor at PD so the tube is
    # well-defined (D = 2*R_SHOCK) at x = 0.  With the orifice, the four
    # orifice break-points already specify D(x) across the PD station;
    # adding (PD_X, 2*R_SHOCK) here would inject a spike back up to the
    # shock-tube diameter at the primary diaphragm and corrupt the
    # geometry seen by L1d.  The reference x2_condition1.py likewise
    # omits any add_break_point at x = 0 when the orifice is present.
    if "x_orifice_centre" not in derived:
        bps.append((PD_X, 2 * R_SHOCK))
    bps.append((L_SHOCK_TUBE_END, 2 * R_SHOCK))
    bps.sort(key=lambda xd: xd[0])
    return bps, derived


# Roberts cluster function (port of gdtk.numeric.roberts for visualisation)
# Used only by plot_geometry() to place cell faces where l1d4-prep will place
# them.  Reproducing this here avoids a runtime gdtk import from the
# optimisation hot path while keeping the cell layout in the plot exact.

def _roberts(eta, alpha, beta):
    """Roberts boundary-layer-like coordinate stretching, eta ∈ [0,1]."""
    lmbda = (beta + 1.0) / (beta - 1.0)
    lmbda = np.power(lmbda, (eta - alpha) / (1.0 - alpha))
    etabar = (beta + 2.0 * alpha) * lmbda - beta + 2.0 * alpha
    return etabar / ((2.0 * alpha + 1.0) * (1.0 + lmbda))


def _distribute_cell_faces(xL, xR, n, end_L, end_R, beta):
    """Return n+1 cell faces between xL and xR.

    Mirrors ``gdtk.numeric.roberts.distribute_points_1`` so the cell
    positions shown by plot_geometry() are bit-identical to what l1d4-prep
    computes for the same slug parameters.  beta < 1 (or no end specified)
    falls back to uniform spacing.
    """
    if ((not end_L) and (not end_R)) or beta < 1.0:
        return np.linspace(xL, xR, n + 1)
    alpha   = 0.0
    reverse = end_L and (not end_R)
    eta     = np.linspace(0.0, 1.0, n + 1)
    if reverse:
        eta = 1.0 - eta
    etabar = _roberts(eta, alpha, beta)
    if reverse:
        etabar = 1.0 - etabar
    return (1.0 - etabar) * xL + etabar * xR


# Geometry visualisation

def plot_geometry(
    buffer_length,
    D_throat=None,
    *,
    mesh_scale=2,
    n_res_base=30,
    n_drv_base=60,
    n_test_base=30,
    drv_cluster_strength=1.01,
    test_cluster_strength=1.02,
    shock_tube_end_x=3.0,
    history_xs=None,
    overview_height_in=1.9,
    save_path=None,
    show=False,
):
    """Render the X2 driver geometry for one (buffer_length, D_throat) pair.

    Two-panel figure:

    * Top - overview of the whole modelled gas path, reservoir start
      anchor to the shock-tube free end.  Drawn with the vertical axis
      exaggerated (see ``overview_height_in``): at true scale the panel
      is ~31:1 and the bore collapses to a hairline.  The exaggeration is
      not stated on the panel, so the caption must carry it.
    * Bottom - zoom on the buffer plate / studs / orifice region, where
      the volume-conserving ramps are too small to resolve at full scale.
      This panel *is* drawn to true 1:1 scale.

    Both panels show:
      * Symmetric area profile (±D/2 vs x), light-grey filled.
      * Break-points as dots below the lower profile (zoom panel only -
        at overview scale they collapse into an unreadable smudge).
      * The piston as a filled dark rectangle at its initial (xL0, xR0)
        position with diameter D_compression.
      * The buffer studs as a hatched rectangle of axial extent
        ``buffer_length`` immediately upstream of BUFFER_PLATE_X_NOMINAL,
        diameter ``D_BUFFER_STUD`` (schematic - radial placement is
        representative, not physical).
      * Gas-slug cell faces as short vertical ticks on the axis and cell
        centres as dots, at the positions L1d will compute via the
        Roberts cluster function (slug colours: reservoir = blue,
        driver = red, test = green).

    Parameters
    ----------
    buffer_length, D_throat
        Same semantics as :func:`build_break_points`.
    mesh_scale
        Multiplier on the per-slug base cell counts; matches
        ``MESH_SCALE_FACTOR`` in l1d_job.py.
    n_res_base, n_drv_base, n_test_base
        Base cell counts for the reservoir, driver, and test gas slugs.
    drv_cluster_strength, test_cluster_strength
        Roberts β for the driver (clustered to PD) and test (clustered
        to PD) slugs.  Pass < 1.0 for uniform spacing.
    overview_height_in
        Height of the overview axes in inches.  Sets how much the top
        panel's vertical axis is stretched relative to true scale; larger
        is more exaggerated.  The zoom panel is unaffected.
    shock_tube_end_x
        Right end of the test-gas slug (== ``right_free.x0`` in the job
        template).  Default mirrors the literal in l1d_job.py; the area
        table runs on to L_SHOCK_TUBE_END but no gas occupies that stretch,
        so this - not L_SHOCK_TUBE_END - is the end of the modelled tube.
    history_xs
        x-positions of the two history locations whose shock arrival
        times give the primary shock speed (time of flight).  Defaults to
        ``TRANSDUCER_XS`` from l1d_job.py, the positions every evaluation
        actually uses.
    save_path
        If set, writes PNG to this path at dpi=150.
    show
        If True, calls ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Rectangle

    if history_xs is None:
        # Imported here, not at module level: l1d_job imports this module,
        # so a top-level import back would be circular.
        from problem.l1d_job import TRANSDUCER_XS
        history_xs = TRANSDUCER_XS

    bps, derived = build_break_points(buffer_length, D_throat)
    xs   = np.array([p[0] for p in bps])
    Ds   = np.array([p[1] for p in bps])
    y_up = +Ds / 2.0
    y_lo = -Ds / 2.0

    piston_xL0   = derived["piston_xL0"]
    piston_xR0   = derived["piston_xR0"]
    pd_x         = derived["pd_x"]
    reservoir_xL = RESERVOIR_START_ANCHOR[0]

    # Cell layouts (matches JOB_SCRIPT_TEMPLATE in l1d_job.py)
    n_res  = n_res_base  * mesh_scale
    n_drv  = n_drv_base  * mesh_scale
    n_test = n_test_base * mesh_scale

    res_faces  = _distribute_cell_faces(
        reservoir_xL, piston_xL0, n_res,
        end_L=False, end_R=False, beta=0.0)
    drv_faces  = _distribute_cell_faces(
        piston_xR0, pd_x, n_drv,
        end_L=False, end_R=True, beta=drv_cluster_strength)
    test_faces = _distribute_cell_faces(
        pd_x, shock_tube_end_x, n_test,
        end_L=True, end_R=False, beta=test_cluster_strength)

    res_ctr  = 0.5 * (res_faces[:-1]  + res_faces[1:])
    drv_ctr  = 0.5 * (drv_faces[:-1]  + drv_faces[1:])
    test_ctr = 0.5 * (test_faces[:-1] + test_faces[1:])

    x_stud_upstr = BUFFER_PLATE_X_NOMINAL - buffer_length

    # Layout
    # Both panels are drawn to scale, but their data aspects differ by
    # more than an order of magnitude (the overview spans ~9.5 m x ~0.5 m,
    # the zoom ~0.3 m x 0.3 m).  A plain subplots() grid gives both cells
    # the same shape, so set_aspect("equal") shrinks each axes inside its
    # cell and the remainder becomes dead margin.  Instead each axes is
    # placed by hand at exactly its own data aspect, and the figure is
    # sized to the result - so there is no intra-axes waste to reclaim.
    # Right edge is the end of the test-gas slug, not the end of the area
    # table: the tube geometry runs on to L_SHOCK_TUBE_END but nothing past
    # the free end is modelled, so the spine itself marks the domain end.
    over_x0, over_x1 = reservoir_xL - 0.3, shock_tube_end_x
    # Left edge must clear whichever feature reaches furthest upstream.
    # The volume-conserving ramp (x_outer_buffer) only creeps upstream a
    # couple of cm over the whole buffer_length range, while the stud
    # block grows with it one-for-one, so at long buffers the studs
    # overtake the ramp and a fixed ramp-relative edge clips them.
    zoom_x0 = min(derived["x_outer_buffer"], x_stud_upstr) - 0.03
    zoom_x1 = (X_ORIFICE_DOWNSTREAM_OUTER + 0.03
               if "x_orifice_centre" in derived else pd_x + 0.02)

    FIG_W, M_L, M_R = 10.0, 0.78, 0.18       # inches
    # PAD_TITLE is 0: both panels are untitled (the caption carries the
    # configuration), so reserving headroom would only add white space.
    PAD_TITLE, PAD_XLABEL = 0.0, 0.50
    # PAD_TOP clears the topmost y-tick label, which straddles the top
    # spine: with the titles gone nothing else holds that space open.
    # PAD_GAP is deliberately generous.  The leader lines run from the
    # indicator box (far right of the overview) down to the zoom's top
    # corners, so at the overview x-label's height each has covered
    # PAD_XLABEL / (PAD_XLABEL + PAD_GAP) of its horizontal travel - the
    # wider the gap, the steeper the lines and the further they stay from
    # the centred label.  At 0.22 they crossed it; 1.2 keeps them clear.
    PAD_GAP, PAD_TOP, PAD_BOT = 1.2, 0.16, 0.08
    ZOOM_H, REGION_FS = 4.0, 9

    ax_w = FIG_W - M_L - M_R

    # The overview is deliberately NOT to scale.  True proportions make it
    # ~19:1, at which the bore is a couple of hairlines thick and the wall
    # transitions are unreadable, so its height is chosen for legibility
    # and the resulting vertical exaggeration is stated on the panel.  The
    # zoom stays at true 1:1 scale.
    over_h = float(overview_height_in)

    # Headroom above the tallest wall for the region labels, which are
    # offset a fixed number of points.  With the height fixed the span and
    # the reservation are mutually dependent, so solve for the span:
    #   span = 2*r_max + foot + head_in * span / over_h
    # which rearranges to the closed form below.
    r_max = float(np.max(y_up))
    head_in = (REGION_FS / 72.0) * 1.45 + 5 / 72.0
    foot_m = 0.02
    over_span = (2 * r_max + foot_m) / (1.0 - head_in / over_h)
    over_y0 = -(r_max + foot_m)
    over_y1 = over_y0 + over_span
    # NB the overview remains vertically exaggerated by roughly x5 (see
    # below); the on-panel note stating so was dropped at the manuscript's
    # request, so the caption must carry the caveat instead.

    zoom_y0, zoom_y1 = -0.15, 0.15           # legend headroom at upper right
    zoom_w = ZOOM_H * (zoom_x1 - zoom_x0) / (zoom_y1 - zoom_y0)
    if zoom_w > ax_w:                        # never overflow the text margins
        zoom_w, ZOOM_H = ax_w, ax_w * (zoom_y1 - zoom_y0) / (zoom_x1 - zoom_x0)

    fig_h = (PAD_BOT + PAD_XLABEL + ZOOM_H + PAD_TITLE + PAD_GAP
             + PAD_XLABEL + over_h + PAD_TITLE + PAD_TOP)
    fig = plt.figure(figsize=(FIG_W, fig_h))

    over_bottom = (PAD_BOT + PAD_XLABEL + ZOOM_H + PAD_TITLE
                   + PAD_GAP + PAD_XLABEL)
    ax_over = fig.add_axes((M_L / FIG_W, over_bottom / fig_h,
                            ax_w / FIG_W, over_h / fig_h))
    # Zoom is far narrower than the overview at equal scale; centring it
    # splits the leftover evenly rather than stranding it on one side.
    ax_zoom = fig.add_axes(((M_L + 0.5 * (ax_w - zoom_w)) / FIG_W,
                            (PAD_BOT + PAD_XLABEL) / fig_h,
                            zoom_w / FIG_W, ZOOM_H / fig_h))

    def _draw_common(ax, break_point_dots=True, equal_aspect=True):
        # Area profile
        ax.fill_between(xs, y_up, y_lo, color="#ececec", lw=0, zorder=0)
        ax.plot(xs, y_up, color="black", lw=1.0, zorder=2)
        ax.plot(xs, y_lo, color="black", lw=1.0, zorder=2)

        # Reservoir back wall.  The upstream end is a closed vessel, so the
        # profile is capped at the first break-point; without it the wall
        # lines just run off the left edge and read as an open duct.  The
        # downstream end is deliberately left open - the shock tube
        # continues past shock_tube_end_x to the dump tank.
        ax.plot([xs[0], xs[0]], [y_lo[0], y_up[0]],
                color="black", lw=1.0, zorder=2)

        # Break-point markers.  Overview omits them: at full scale most of
        # the break-points fall within a few cm of the primary diaphragm
        # and merge into one smudge, so they only read on the zoom.
        if break_point_dots:
            ax.plot(xs, y_lo, "o", ms=4, color="black", zorder=3)

        # Piston
        ax.add_patch(Rectangle(
            (piston_xL0, -R_COMPRESSION),
            piston_xR0 - piston_xL0, 2 * R_COMPRESSION,
            facecolor="#404040", edgecolor="black", lw=0.8, zorder=4,
        ))

        # Buffer studs (axial extent + cross-section; radial position
        # schematic - see docstring).
        if buffer_length > 0:
            ax.add_patch(Rectangle(
                (x_stud_upstr, -D_BUFFER_STUD / 2.0),
                buffer_length, D_BUFFER_STUD,
                facecolor="none", edgecolor="black",
                lw=0.8, hatch="////", zorder=4,
            ))

        # Cell faces (short axial ticks) + cell centres (dots)
        tk = 0.005   # 5 mm half-tick
        for xf in res_faces:
            ax.plot([xf, xf], [-tk, +tk],
                    color="steelblue", lw=0.4, alpha=0.55, zorder=1)
        for xf in drv_faces:
            ax.plot([xf, xf], [-tk, +tk],
                    color="crimson",  lw=0.4, alpha=0.55, zorder=1)
        for xf in test_faces:
            ax.plot([xf, xf], [-tk, +tk],
                    color="darkgreen", lw=0.4, alpha=0.55, zorder=1)
        ax.plot(res_ctr,  np.zeros_like(res_ctr),  ".",
                ms=2, color="steelblue", alpha=0.55, zorder=1)
        ax.plot(drv_ctr,  np.zeros_like(drv_ctr),  ".",
                ms=2, color="crimson",   alpha=0.55, zorder=1)
        ax.plot(test_ctr, np.zeros_like(test_ctr), ".",
                ms=2, color="darkgreen", alpha=0.55, zorder=1)

        ax.set_xlabel("x-location (m)")
        ax.set_ylabel("Radius (m)")
        ax.grid(True, lw=0.3, alpha=0.4)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

    _draw_common(ax_over, break_point_dots=False, equal_aspect=False)
    _draw_common(ax_zoom, break_point_dots=True, equal_aspect=True)

    # Region labels on the overview only (cluttered on zoom).  Each sits
    # just clear of its own local wall rather than inside the bore, where
    # the cell-face ticks and centre dots ran through the text.  The
    # profile is monotonic in x, so np.interp gives the local wall height.
    # Offset in points rather than data units: the overview is ~19x wider
    # than it is tall, so a fixed data-unit pad would sit a wildly
    # different physical distance above the wall than the same pad does on
    # the zoom, and would rescale whenever the figure width changes.
    def _label_above(ax, x_mid, text, dy=5):
        ax.annotate(text, xy=(x_mid, float(np.interp(x_mid, xs, y_up))),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=REGION_FS)

    _label_above(ax_over, 0.5 * (reservoir_xL + piston_xL0), "Reservoir")
    _label_above(ax_over, 0.5 * (piston_xR0 + pd_x), "Compression Tube")
    _label_above(ax_over, piston_xL0 + 0.5 * (piston_xR0 - piston_xL0),
                 "Piston")
    _label_above(ax_over, 0.5 * (pd_x + shock_tube_end_x), "Shock Tube")

    # Time-of-flight history locations.  Drawn as bars across the bore
    # (where the probe samples) rather than full-height lines, which would
    # run through the "Shock Tube" label sitting above station 1.
    ax_over.vlines(history_xs, -R_SHOCK, R_SHOCK, color="tab:purple",
                   lw=1.6, zorder=5,
                   label="ToF history locations (x = "
                         + ", ".join(f"{x:g}" for x in history_xs)
                         + " m)")
    # Upper right: clear air above the narrow shock tube.  Lower right put
    # the box over the end of the compression-tube wall.
    ax_over.legend(loc="upper right", fontsize=8)

    # Overview limits - match Hodson Fig 4.5 framing.  y-limits are the
    # ones the layout block sized the axes from; changing them here would
    # break the equal-aspect fit.
    ax_over.set_xlim(over_x0, over_x1)
    ax_over.set_ylim(over_y0, over_y1)

    # Zoom - buffer plate + orifice region.  Right edge sized to show
    # the full orifice plate (the expansion ramp now extends past the
    # PD to X_ORIFICE_DOWNSTREAM_FACE + a few mm of headroom).
    ax_zoom.set_xlim(zoom_x0, zoom_x1)
    ax_zoom.set_ylim(zoom_y0, zoom_y1)

    # Inset indicator: mark the zoomed span on the overview and run leader
    # lines down to the zoom panel, so the reader can place the detail.
    # The box is keyed to the zoom's own limits rather than to zoom_x0 /
    # zoom_y0 directly, so it cannot drift from what the panel shows.
    ind_kw = dict(color="0.25", lw=0.8, ls="--")
    # edgecolor, not color: on a patch, color sets both edge and face and
    # would override facecolor="none", filling the box over the tube.
    ax_over.add_patch(Rectangle(
        (zoom_x0, zoom_y0), zoom_x1 - zoom_x0, zoom_y1 - zoom_y0,
        facecolor="none", edgecolor=ind_kw["color"],
        lw=ind_kw["lw"], ls=ind_kw["ls"], zorder=6,
    ))
    # Leader lines span two axes, which no single axes' coordinates can
    # express: each end gets its own transform - overview data coords at
    # the box's lower corners, zoom axes-fraction at the panel's top
    # corners.  Held by the overview (not the figure, which would draw
    # last and so over everything) so that the zoom panel and its title,
    # drawn after it, cover the lines where they cross.
    for x_box, x_frac in ((zoom_x0, 0.0), (zoom_x1, 1.0)):
        ax_over.add_artist(ConnectionPatch(
            xyA=(x_box, zoom_y0), coordsA=ax_over.transData,
            xyB=(x_frac, 1.0), coordsB=ax_zoom.transAxes,
            zorder=1.5, clip_on=False, **ind_kw,
        ))
    ax_zoom.axvline(pd_x, color="orange", ls="--", lw=1.8, alpha=0.7,
                    label="Primary diaphragm (PD)")

    # Displaced stud volume (Eq 4.13) - this is the V_buffer that sets the
    # upstream volume-conserving ramp, so it belongs beside the geometry it
    # produces.  Carried by the stud patch's own legend entry rather than a
    # free-floating text box, so it stays attached to the hatched region.
    V_buffer = buffer_stud_volume(buffer_length)
    handles, labels = ax_zoom.get_legend_handles_labels()
    if buffer_length > 0:
        # Proxy artist: built for the legend only and deliberately never
        # added to the axes, so it carries the hatch swatch without being
        # drawn on the plot.  (Adding it and hiding it suppresses the
        # legend handle too.)
        handles.append(Rectangle((0, 0), 1, 1, facecolor="none",
                                 edgecolor="black", lw=0.8, hatch="////"))
        labels.append(
            f"Buffer studs, $V_\\mathrm{{buffer}}$ = {V_buffer * 1e3:.3f} L"
        )
    ax_zoom.legend(handles, labels, loc="upper right", fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


# Self-test - reproduces Table 4.5 (basic config, no studs, no orifice)

if __name__ == "__main__":
    bps, derived = build_break_points(buffer_length=0.0, D_throat=None)

    # Hodson Table 4.5, rounded to four decimals as in the paper.
    expected = [
        (-8.7188, 0.3160),
        (-5.8093, 0.3160),
        (-5.7913, 0.2440),
        (-5.1893, 0.2440),
        (-5.1733, 0.1800),
        (-5.1436, 0.1800),
        (-5.1376, 0.1561),
        (-4.8332, 0.1561),
        (-4.8080, 0.2568),
        (-4.5870, 0.2568),
        (-0.1299, 0.2568),
        (-0.0869, 0.0850),
        ( 0.0000, 0.0850),
        ( 5.0000, 0.0850),
    ]

    print(f"Generated {len(bps)} break-points; expected {len(expected)}.")
    if len(bps) != len(expected):
        raise SystemExit(f"FAIL: length mismatch")

    for i, ((x_got, D_got), (x_exp, D_exp)) in enumerate(zip(bps, expected)):
        match = abs(x_got - x_exp) < 5e-4 and abs(D_got - D_exp) < 1e-4
        flag = "OK" if match else "FAIL"
        print(f"  [{i:2d}] x={x_got:+8.4f} (exp {x_exp:+8.4f})  "
              f"D={D_got:.4f} (exp {D_exp:.4f})  {flag}")
        if not match:
            raise SystemExit("Table 4.5 mismatch - methodology not reproduced")

    # Orifice sanity check.
    #
    # Reference matches required:
    #   * D_throat = 0.065: match x2_condition1.py (original) exactly.
    #   * D_throat = 0.077: match x2_condition1.py (D=77mm variant)
    #     to within ~16 microns (rounding tolerance in reference).
    # At smaller D_throat the upstream chamfer keeps slope 1 and grows
    # outward; the downstream endpoints stay fixed and its slope
    # flattens.  At D_throat == 2*R_SHOCK the orifice is degenerate
    # and the block is skipped.
    expected_orifice = {
        0.065: [
            (-0.014778, 0.085),
            (-0.004778, 0.065),
            (+0.100000, 0.065),
            (+0.110000, 0.085),
        ],
        0.077: [
            (-0.011967, 0.085),
            (-0.007967, 0.077),
            (+0.100000, 0.077),
            (+0.110000, 0.085),
        ],
    }
    for D_th in (0.085, 0.077, 0.065, 0.05, 0.04):
        bps_o, derived_o = build_break_points(0.10, D_throat=D_th)
        has_orifice = "x_orifice_centre" in derived_o
        print(f"\nOrifice config: D_throat={D_th:.3f} m, buffer_length=0.10 m")
        print(f"  total break-points = {len(bps_o)}  (orifice inserted: {has_orifice})")
        print(f"  x_inner_buffer     = {derived_o['x_inner_buffer']:+.5f}")
        if not has_orifice:
            continue
        print(f"  x_orifice_centre   = {derived_o['x_orifice_centre']:+.5f}")

        # Recover the four orifice break-points: the two throat
        # (D == D_throat) points plus the shock-D points immediately
        # upstream and downstream of them.
        throat_bps = [bp for bp in bps_o if abs(bp[1] - D_th) < 1e-9]
        i_contr_throat = bps_o.index(throat_bps[0])
        i_expan_throat = bps_o.index(throat_bps[-1])
        upstream_shock   = bps_o[i_contr_throat - 1]
        downstream_shock = bps_o[i_expan_throat + 1]
        orifice_bps = [upstream_shock, throat_bps[0],
                       throat_bps[-1], downstream_shock]
        throat_span = orifice_bps[2][0] - orifice_bps[1][0]
        ramp_up_len = orifice_bps[1][0] - orifice_bps[0][0]
        ramp_dn_len = orifice_bps[3][0] - orifice_bps[2][0]
        print(f"  contraction ramp ({orifice_bps[0][0]:+.6f}, {orifice_bps[0][1]:.4f}) "
              f"-> ({orifice_bps[1][0]:+.6f}, {orifice_bps[1][1]:.4f}) "
              f"[len {ramp_up_len*1e3:.3f} mm]")
        print(f"  throat span      {throat_span * 1e3:.3f} mm "
              f"({orifice_bps[1][0]:+.6f} -> {orifice_bps[2][0]:+.6f})")
        print(f"  expansion ramp   ({orifice_bps[2][0]:+.6f}, {orifice_bps[2][1]:.4f}) "
              f"-> ({orifice_bps[3][0]:+.6f}, {orifice_bps[3][1]:.4f}) "
              f"[len {ramp_dn_len*1e3:.3f} mm]")

        if D_th in expected_orifice:
            if abs(ORIFICE_X_SHIFT) < 1e-12:
                print("  Checking against reference (x2_condition1.py):")
                # 20 micron tolerance covers the rounding in the D=77mm
                # reference file (~16 micron deltas on the upstream side).
                tol = 2e-5
                for got, exp in zip(orifice_bps, expected_orifice[D_th]):
                    ok = abs(got[0] - exp[0]) < tol and abs(got[1] - exp[1]) < 1e-6
                    flag = "OK" if ok else "FAIL"
                    print(f"    got=({got[0]:+.6f}, {got[1]:.4f})  "
                          f"exp=({exp[0]:+.6f}, {exp[1]:.4f})  {flag}")
                    if not ok:
                        raise SystemExit(
                            f"Reference orifice mismatch at D_throat={D_th}"
                        )
            else:
                print(f"  Reference comparison skipped: ORIFICE_X_SHIFT="
                      f"{ORIFICE_X_SHIFT:+.4f} m (deliberate divergence "
                      f"from reference for PD-throat margin).")
                for got, exp in zip(orifice_bps, expected_orifice[D_th]):
                    delta_x = got[0] - exp[0]
                    print(f"    got=({got[0]:+.6f}, {got[1]:.4f})  "
                          f"exp=({exp[0]:+.6f}, {exp[1]:.4f})  "
                          f"dx={delta_x*1e3:+.3f} mm")

    # Buffer-plate cross-check.  The new buffer = 130 mm reference file
    # (no orifice) gives buffer-plate break-points (-0.163085, 0.2568)
    # and (-0.120135, 0.085).  Verify our volume_conserving_ramp +
    # buffer_stud_volume reproduces those positions as well.
    expected_buffer_at_130 = [(-0.163085, 0.2568), (-0.120135, 0.085)]
    bps_b, derived_b = build_break_points(buffer_length=0.130, D_throat=None)
    # The buffer-plate ramp lives between the compression-tube anchor
    # (PISTON_FRONT_X, D=2*R_COMPRESSION) and the first shock-tube-D
    # anchor downstream of it.
    i_pf = bps_b.index((PISTON_FRONT_X, 2 * R_COMPRESSION))
    buffer_bps = [bps_b[i_pf + 1], bps_b[i_pf + 2]]
    print("\nBuffer plate check at buffer_length = 0.130 m:")
    tol = 2e-5
    for got, exp in zip(buffer_bps, expected_buffer_at_130):
        ok = abs(got[0] - exp[0]) < tol and abs(got[1] - exp[1]) < 1e-6
        flag = "OK" if ok else "FAIL"
        print(f"  got=({got[0]:+.6f}, {got[1]:.4f})  "
              f"exp=({exp[0]:+.6f}, {exp[1]:.4f})  {flag}")
        if not ok:
            raise SystemExit("Buffer-plate mismatch at buffer_length=0.130")

    print("\nAll smoke tests passed.")
