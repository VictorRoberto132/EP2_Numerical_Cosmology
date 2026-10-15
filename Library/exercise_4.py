"""
Exercise 4 - Spherical collapse with three numerical formulations (A, B, C).

Smooth-dark-energy, shear-free limit of Pace, Meyer & Bartelmann (2017).
Reference: Lecture 8, notes Sections 3 and 6-7.

Conventions:
    N = ln a,  X_N = dX/dN
    A_H   = 2 + d ln H / dN
    theta = (div_x v) / (a H)       (dimensionless velocity divergence)
    g     = 1 / delta               (inverse overdensity, NOT the growth rate)
    f_i   = d ln D / d ln a at a_i  (growth rate, from Exercise 1)

Background: flat LCDM (OL0 = 1 - Om0), radiation neglected. EdS is Om0 = 1.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import exercise_1 as ex1


# ---------------------------------------------------------------------------
# Background (flat, matter + Lambda). We reuse the Exercise 1 functions.
# ---------------------------------------------------------------------------

def A_H(a, Om0):
    """A_H = 2 + d ln H/dN. Flat LCDM: 2 - (3/2) Omega_m(a); EdS: 1/2 (notes eq. 6.1)."""
    return 2.0 + ex1.dlnH_dN(a, Om0, 1.0 - Om0)


def Omega_m(a, Om0):
    """Omega_m(a) for flat LCDM."""
    return ex1.Omega_m(a, Om0, 1.0 - Om0)


def Omega_L(a, Om0):
    """Omega_Lambda(a) = 1 - Omega_m(a) for flat LCDM (no radiation, no curvature)."""
    return 1.0 - Omega_m(a, Om0)


# ---------------------------------------------------------------------------
# Right-hand sides of the three nonlinear systems and of the linear equation.
# Each one is written from its own equation (notes eqs. 6.2-6.5).
# ---------------------------------------------------------------------------

def rhs_A(N, y, Om0):
    """Formulation A (eq. 6.2) as a first-order system y = [delta, delta_N]."""
    delta, delta_N = y
    a = np.exp(N)
    delta_NN = (-A_H(a, Om0) * delta_N
                + (4.0 / 3.0) * delta_N**2 / (1.0 + delta)
                + 1.5 * Omega_m(a, Om0) * delta * (1.0 + delta))
    return [delta_N, delta_NN]


def rhs_B(N, y, Om0):
    """Formulation B (eq. 6.4), y = [delta, theta]."""
    delta, theta = y
    a = np.exp(N)
    delta_N = -(1.0 + delta) * theta
    theta_N = -A_H(a, Om0) * theta - theta**2 / 3.0 - 1.5 * Omega_m(a, Om0) * delta
    return [delta_N, theta_N]


def rhs_C(N, y, Om0):
    """Formulation C (eq. 6.5), y = [g, theta] with g = 1/delta."""
    g, theta = y
    a = np.exp(N)
    g_N = g * (1.0 + g) * theta
    theta_N = -A_H(a, Om0) * theta - theta**2 / 3.0 - 1.5 * Omega_m(a, Om0) / g
    return [g_N, theta_N]


def rhs_linear(N, y, Om0):
    """Linear counterpart (eq. 6.3), y = [delta_L, delta_L_N]."""
    dL, dL_N = y
    a = np.exp(N)
    dL_NN = -A_H(a, Om0) * dL_N + 1.5 * Omega_m(a, Om0) * dL
    return [dL_N, dL_NN]


RHS = {"A": rhs_A, "B": rhs_B, "C": rhs_C}


# ---------------------------------------------------------------------------
# Initial conditions (growing mode, notes eq. 6.6)
# ---------------------------------------------------------------------------

def growth_rate_initial(a_i, Om0):
    """
    f_i = d ln D / d ln a at a_i, from the Exercise 1 growth ODE.
    The growth ODE is started 100 times earlier than a_i, so that f_i is not
    simply the value imposed by its own initial condition (G = G_N).
    """
    _, f = ex1.growth_ode([a_i], Om0, 1.0 - Om0, a_i=a_i / 100.0)
    return f[0]


def initial_state(form, delta_i, f_i):
    """Initial state vector of formulation A, B or C for a trial delta_i."""
    if form == "A":
        return [delta_i, f_i * delta_i]                       # [delta, delta_N]
    if form == "B":
        return [delta_i, -f_i * delta_i / (1.0 + delta_i)]    # [delta, theta]
    if form == "C":
        return [1.0 / delta_i, -f_i * delta_i / (1.0 + delta_i)]  # [g, theta]
    raise ValueError("form must be 'A', 'B' or 'C'")


# ---------------------------------------------------------------------------
# Events: collapse (terminal) and turnaround (theta = -3, not terminal)
# ---------------------------------------------------------------------------

def make_events(form, threshold):
    """
    Return (collapse_event, turnaround_event) for solve_ivp.

    threshold = delta_inf for A and B (event delta = delta_inf),
              = eps       for C       (event g = eps > 0).
    """
    if form in ("A", "B"):
        def collapse(N, y, Om0):
            return y[0] - threshold          # delta - delta_inf crosses 0 upwards
        collapse.direction = 1
    else:
        def collapse(N, y, Om0):
            return y[0] - threshold          # g - eps crosses 0 downwards
        collapse.direction = -1
    collapse.terminal = True                 # stop: never integrate across the singularity

    def turnaround(N, y, Om0):
        # theta for each formulation (A reconstructs it: theta = -delta_N / (1 + delta))
        theta = -y[1] / (1.0 + y[0]) if form == "A" else y[1]
        return theta + 3.0                   # turnaround: d ln R / dN = 1 + theta/3 = 0
    turnaround.direction = -1                # theta decreases through -3
    turnaround.terminal = False

    return collapse, turnaround


def default_threshold(form):
    """Default event threshold: delta_inf = 1e8 (A, B) or eps = 1e-8 (C)."""
    return 1e8 if form in ("A", "B") else 1e-8


# ---------------------------------------------------------------------------
# One nonlinear integration for a given delta_i
# ---------------------------------------------------------------------------

def integrate_nonlinear(form, delta_i, Om0, a_i, N_end, threshold, rtol, atol,
                        method="DOP853", dense_output=False):
    """
    Integrate formulation `form` from N_i = ln a_i up to N_end, or up to the
    collapse event if it happens first. Returns the solve_ivp solution.
    """
    f_i = growth_rate_initial(a_i, Om0)
    y0 = initial_state(form, delta_i, f_i)
    collapse, turnaround = make_events(form, threshold)
    return solve_ivp(RHS[form], (np.log(a_i), N_end), y0, args=(Om0,),
                     events=[collapse, turnaround], method=method,
                     rtol=rtol, atol=atol, dense_output=dense_output)


def collapse_N(form, delta_i, Om0, a_i, N_end, threshold, rtol, atol, method="DOP853"):
    """N = ln a at the collapse event, or N_end if the sphere has not collapsed by then."""
    sol = integrate_nonlinear(form, delta_i, Om0, a_i, N_end, threshold, rtol, atol, method)
    N_coll = sol.t_events[0]
    return N_coll[0] if len(N_coll) > 0 else N_end


def linear_delta(delta_i, Om0, a_i, a_eval, rtol, atol, method="DOP853"):
    """Linear solution delta_L(a) with delta_L,i = delta_i and delta_L,N,i = f_i delta_i."""
    f_i = growth_rate_initial(a_i, Om0)
    a_eval = np.atleast_1d(a_eval)
    sol = solve_ivp(rhs_linear, (np.log(a_i), np.log(a_eval.max())),
                    [delta_i, f_i * delta_i], args=(Om0,), t_eval=np.log(a_eval),
                    method=method, rtol=rtol, atol=atol)
    return sol.y[0]


# ---------------------------------------------------------------------------
# Virial prescription (notes Section 3.3, eqs. 3.13-3.15) - common to A, B, C
# ---------------------------------------------------------------------------

def virial_y(lam):
    """
    Root y = R_vir/R_ta of 4 lam y^3 - 2 (1 + lam) y + 1 = 0 connected to y = 1/2 as lam -> 0.
    The cubic is +1 at y = 0 and -lam/2 <= 0 at y = 1/2, so this root lies in (0, 1/2].
    """
    if lam == 0.0:
        return 0.5                                   # EdS
    cubic = lambda y: 4.0 * lam * y**3 - 2.0 * (1.0 + lam) * y + 1.0
    return brentq(cubic, 0.0, 0.5, xtol=1e-15)


def virial_overdensities(a_ta, zeta_ta, a_c, Om0):
    """
    Delta_vir,m and Delta_vir,c at the formal collapse epoch a_c (eq. 3.15).
    lambda_ta = rho_Lambda / rho_sph,ta = Omega_L(a_ta) / (Omega_m(a_ta) zeta_ta) (eq. 3.13).
    """
    lam = Omega_L(a_ta, Om0) / (Omega_m(a_ta, Om0) * zeta_ta)
    y = virial_y(lam)
    Dvir_m = zeta_ta * (a_c / a_ta)**3 / y**3
    Dvir_c = Omega_m(a_c, Om0) * Dvir_m
    return Dvir_m, Dvir_c, y, lam


# ---------------------------------------------------------------------------
# Full calculation: shooting + turnaround + delta_c + virial
# ---------------------------------------------------------------------------

def spherical_collapse(form, z_c, Om0, a_i=1e-5, threshold=None,
                       rtol=1e-10, atol=1e-14, method="DOP853", xtol_rel=1e-13):
    """
    Spherical collapse for one formulation, cosmology and target redshift.

    Parameters
    ----------
    form : 'A', 'B' or 'C'.
    z_c : target collapse redshift, a_c = 1/(1 + z_c).
    Om0 : matter density today (flat: OL0 = 1 - Om0). Om0 = 1 is EdS.
    a_i : initial scale factor (a_i << a_c).
    threshold : delta_inf (A, B) or eps (C). Default 1e8 or 1e-8.
    rtol, atol : solve_ivp tolerances (used for the nonlinear and linear ODEs).
    method : solve_ivp integrator.
    xtol_rel : relative tolerance of the root search on delta_i.

    Returns
    -------
    dict with the tuned delta_i, turnaround (a_ta, zeta_ta), delta_c, the virial
    overdensities, the root-search bracket, the event residual and the settings.
    """
    if threshold is None:
        threshold = default_threshold(form)

    a_c = 1.0 / (1.0 + z_c)
    N_c = np.log(a_c)
    # Integrate at most up to a = 2 a_c: trial spheres that do not collapse by then
    # simply return N_end (the root function is then positive, as it should be).
    N_end = N_c + np.log(2.0)

    # Root function: (collapse epoch) - (target epoch), in N. It decreases with delta_i.
    def F(delta_i):
        return collapse_N(form, delta_i, Om0, a_i, N_end, threshold, rtol, atol, method) - N_c

    # Initial guess from linear theory: delta_L(a_c) = 1.686 with delta_L proportional to D.
    D = ex1.growth_ode([a_i, a_c], Om0, 1.0 - Om0, a_i=a_i / 100.0)[0]
    guess = 1.686 * D[0] / D[1]

    # Bracket [lo, hi] such that F(lo) > 0 (collapses too late) and F(hi) < 0 (too early)
    lo, hi = 0.9 * guess, 1.1 * guess
    while F(lo) <= 0:
        lo *= 0.9
    while F(hi) >= 0:
        hi *= 1.1

    # Bracketed root search (Brent) for the tuned initial amplitude
    delta_i, info = brentq(F, lo, hi, xtol=xtol_rel * guess, rtol=4 * np.finfo(float).eps,
                           full_output=True)

    # Final run with the tuned delta_i: collapse event and turnaround event
    sol = integrate_nonlinear(form, delta_i, Om0, a_i, N_end, threshold, rtol, atol, method)
    N_coll = sol.t_events[0][0]
    N_ta = sol.t_events[1][0]
    y_ta = sol.y_events[1][0]

    # Turnaround density: C reconstructs delta = 1/g
    delta_ta = 1.0 / y_ta[0] if form == "C" else y_ta[0]
    a_ta = np.exp(N_ta)
    zeta_ta = 1.0 + delta_ta

    # Linear threshold delta_c = delta_L(a_c) with the same delta_i
    delta_c = linear_delta(delta_i, Om0, a_i, a_c, rtol, atol, method)[0]

    # Common virial prescription
    Dvir_m, Dvir_c, y, lam = virial_overdensities(a_ta, zeta_ta, a_c, Om0)

    return {"form": form, "z_c": z_c, "Om0": Om0, "a_i": a_i, "threshold": threshold,
            "rtol": rtol, "atol": atol, "method": method,
            "delta_i": delta_i, "bracket": (lo, hi), "n_iter": info.iterations,
            "event_residual_N": N_coll - N_c,          # ln(a_event / a_c)
            "a_ta": a_ta, "a_ta/a_c": a_ta / a_c, "zeta_ta": zeta_ta,
            "delta_c": delta_c, "y": y, "lambda_ta": lam,
            "Dvir_m": Dvir_m, "Dvir_c": Dvir_c}
