"""
Exercise 1 - Linear growth.

Late-time background with pressureless matter, curvature and a cosmological
constant (no radiation). Reference: Lecture 5, notes Sections 3.4-3.7.

Time variable: N = ln a, and X_N = dX/dN.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad


# ---------------------------------------------------------------------------
# Background (Lecture 5, eqs. 3.25-3.26)
# ---------------------------------------------------------------------------

def E2(a, Om0, OL0):
    """E^2(a) = H^2/H0^2 = Om0 a^-3 + Ok0 a^-2 + OL0, with Ok0 = 1 - Om0 - OL0."""
    Ok0 = 1.0 - Om0 - OL0
    return Om0 * a**-3 + Ok0 * a**-2 + OL0


def Omega_m(a, Om0, OL0):
    """Matter density parameter at scale factor a: Om0 a^-3 / E^2(a)."""
    return Om0 * a**-3 / E2(a, Om0, OL0)


def dlnH_dN(a, Om0, OL0):
    """
    d ln H / dN = (1 / 2E^2) dE^2/dN.
    Since dE^2/dN = -3 Om0 a^-3 - 2 Ok0 a^-2 (OL0 is constant),
    this is computed analytically (no numerical derivative).
    """
    Ok0 = 1.0 - Om0 - OL0
    dE2_dN = -3.0 * Om0 * a**-3 - 2.0 * Ok0 * a**-2
    return 0.5 * dE2_dN / E2(a, Om0, OL0)


# ---------------------------------------------------------------------------
# Growing-mode equation (Lecture 5, eq. 3.24)
#   G_NN + (2 + dlnH/dN) G_N - (3/2) Omega_m(a) G = 0
# ---------------------------------------------------------------------------

# rhs = right-hand side of the growth ODE written as a first-order system.
def growth_rhs(N, y, Om0, OL0):
    """Right-hand side of the growth ODE written as a first-order system y = [G, G_N]."""
    G, G_N = y
    a = np.exp(N)
    G_NN = -(2.0 + dlnH_dN(a, Om0, OL0)) * G_N + 1.5 * Omega_m(a, Om0, OL0) * G
    return [G_N, G_NN]


def growth_ode(a, Om0, OL0, a_i=1e-3, rtol=1e-10, atol=1e-12):
    """
    Solve the growth ODE in N = ln a and return D(a) and f(a).

    Parameters
    ----------
    a : array of scale factors where D and f are returned (a_i <= a).
    Om0, OL0 : matter and cosmological-constant density parameters today.
               Curvature is Ok0 = 1 - Om0 - OL0.
    a_i : initial scale factor, deep in matter domination.
    rtol, atol : solver tolerances.

    Returns
    -------
    D : G(a) / G(1), growth factor normalized to 1 today.
    f : G_N / G, growth rate (taken directly from the ODE state, not from
        differentiating an interpolated D).
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))

    # Integrate from N_i up to today (or further, if later a are asked for)
    N_i = np.log(a_i)
    N_end = np.log(max(1.0, a.max()))

    # Growing-mode initial conditions (eq. 3.28): G(a_i) = G_N(a_i) = a_i
    y0 = [a_i, a_i]

    # Evaluate at the requested points and also at a = 1 (N = 0) for the normalization.
    # np.unique sorts and removes duplicates (solve_ivp needs strictly increasing t_eval);
    # 'inverse' maps each entry back to its original position.
    N_eval, inverse = np.unique(np.append(np.log(a), 0.0), return_inverse=True)

    sol = solve_ivp(growth_rhs, (N_i, N_end), y0, args=(Om0, OL0),
                    t_eval=N_eval, method="RK45", rtol=rtol, atol=atol)

    # Back to the original order; the last entry corresponds to a = 1
    G = sol.y[0][inverse]
    G_N = sol.y[1][inverse]
    G_today = G[-1]
    G, G_N = G[:-1], G_N[:-1]

    D = G / G_today
    f = G_N / G
    return D, f


# ---------------------------------------------------------------------------
# Integral growing solution (Lecture 5, eq. 3.33)
#   G(a) = (5 Om0 / 2) E(a) int_0^a da' / (a'^3 E^3(a'))
# Exact for matter + curvature + cosmological constant.
# ---------------------------------------------------------------------------

def growth_integrand(a, Om0, OL0):
    """Integrand 1 / (a^3 E^3(a)). It behaves as a^(3/2) / Om0^(3/2) for a -> 0."""
    return 1.0 / (a**3 * E2(a, Om0, OL0)**1.5)


def growth_quad(a, Om0, OL0, epsrel=1e-12, epsabs=0.0):
    """
    Growth factor from the quadrature, independent of the ODE.

    Parameters
    ----------
    a : array of scale factors where D is returned.
    Om0, OL0 : matter and cosmological-constant density parameters today.
    epsrel, epsabs : tolerances passed to scipy.integrate.quad.

    Returns
    -------
    D : G(a) / G(1), growth factor normalized to 1 today.
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))

    def G(a_val):
        # quad uses interior points only, so the integrand is never evaluated at a = 0
        integral, _ = quad(growth_integrand, 0.0, a_val, args=(Om0, OL0),
                           epsrel=epsrel, epsabs=epsabs)
        return 2.5 * Om0 * np.sqrt(E2(a_val, Om0, OL0)) * integral

    # Evaluate G at each requested a, then normalize by G(1)
    G_a = np.array([G(a_val) for a_val in a])
    return G_a / G(1.0)
