"""
Exercise 5 - Validate the primordial-to-late-time numerical map.

Reference: Lecture 9, notes Sections 6 and 8.4-8.6.

Tools:
    CLASS v3.4.0 (commit 64bbab7), C executable, run from .ini files.
    CAMB (python package), for the independent-code comparison.

Conventions:
    CLASS tables:  kh in h/Mpc, Ph in (Mpc/h)^3   ->   k = h kh [1/Mpc],  P = Ph / h^3 [Mpc^3]
    Transfer:      T_m = delta_m / R  (CLASS 'd_m', normalized to initial curvature R = 1)
    Primordial:    Delta_R^2(k) = A_s (k / k_*)^(n_s - 1)
    Reconstruction P_rec(k) = (2 pi^2 / k^3) Delta_R^2(k) T_m^2(k)        (notes eq. 8.4)
"""

import os
import re
import subprocess

import numpy as np
from scipy.interpolate import CubicSpline

import camb


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# CLASS executable (built from the v3.4.0 tag). Can be overridden with $CLASS_EXE.
CLASS_EXE = os.environ.get(
    "CLASS_EXE",
    "/home/vroberto/miniconda3/envs/skewerjax_fakespectra/src/class_public/class")

# Commit of the CLASS source used (tag v3.4.0), quoted in the notes, Section 8.5
CLASS_COMMIT = "64bbab707faf4de4779a9e04edd180fef18d98fa"

# Working copy of the fiducial configuration (next to this file)
BASE_INI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "exercise_5_class_base.ini")


# ---------------------------------------------------------------------------
# .ini files and CLASS runs
# ---------------------------------------------------------------------------

def read_ini(path=BASE_INI):
    """Read a CLASS .ini file into a dict {name: value-as-string} (comments skipped)."""
    params = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()      # drop comments
            if "=" in line:
                key, val = line.split("=", 1)
                params[key.strip()] = val.strip()
    return params


def write_ini(params, path):
    """Write a dict {name: value} as a CLASS .ini file."""
    with open(path, "w") as f:
        for key, val in params.items():
            f.write(f"{key} = {val}\n")


def run_class(params, name, out_dir, rerun=False):
    """
    Run the CLASS executable for the parameters 'params' (dict).

    Files written in out_dir:
        <name>.ini              the exact input passed to CLASS
        <name>.log              CLASS standard output (derived quantities)
        <name>_tk.dat, <name>_pk.dat, ...   CLASS output tables

    If rerun=False and the tables already exist, CLASS is not run again.
    Returns the root path out_dir/<name>_ .
    """
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.join(os.path.abspath(out_dir), name + "_")

    if rerun or not os.path.exists(root + "pk.dat"):
        p = dict(params)
        p["root"] = root[:-1]                   # CLASS appends the "_" itself
        p["overwrite_root"] = "yes"
        write_ini(p, root[:-1] + ".ini")
        out = subprocess.run([CLASS_EXE, root[:-1] + ".ini"],
                             capture_output=True, text=True)
        with open(root[:-1] + ".log", "w") as f:
            f.write(out.stdout + out.stderr)
        if out.returncode != 0:
            raise RuntimeError(f"CLASS failed for {name}:\n{out.stdout[-2000:]}")
    return root


def read_class_table(filename):
    """Read a CLASS output table. Returns a dict {column name: array}."""
    names = None
    with open(filename) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if "1:" in line:                          # header line with the column names
                names = re.findall(r"\d+:(.+?)(?=\s+\d+:|$)", line[1:].strip())
                names = [n.strip() for n in names]
    data = np.loadtxt(filename)
    return {n: data[:, i] for i, n in enumerate(names)}


def read_log_value(root, pattern):
    """Read one number from the CLASS run log with a regular expression."""
    with open(root[:-1] + ".log") as f:
        match = re.search(pattern, f.read())
    return float(match.group(1))


def class_linear_pk(root, h):
    """
    Linear total-matter P(k) of a CLASS run, converted to k [1/Mpc] and P [Mpc^3].
    """
    pk = read_class_table(root + "pk.dat")
    k = h * pk["k (h/Mpc)"]
    P = pk["P (Mpc/h)^3"] / h**3
    return k, P


# ---------------------------------------------------------------------------
# (a) Reconstruction from the transfer function
# ---------------------------------------------------------------------------

def primordial_spectrum(k, A_s, n_s, k_pivot):
    """Dimensionless curvature spectrum Delta_R^2(k) = A_s (k/k_*)^(n_s-1), k in 1/Mpc."""
    return A_s * (k / k_pivot)**(n_s - 1.0)


def reconstruct_pk(k, T_m, A_s, n_s, k_pivot):
    """P_rec(k) = (2 pi^2/k^3) Delta_R^2(k) T_m^2, k in 1/Mpc, P in Mpc^3 (notes eq. 8.4)."""
    return 2.0 * np.pi**2 / k**3 * primordial_spectrum(k, A_s, n_s, k_pivot) * T_m**2


# ---------------------------------------------------------------------------
# (b) CLASS precision controls
# ---------------------------------------------------------------------------

# Tighter reference settings. Hierarchy depths and approximation switches are
# those of CLASS's pk_ref.pre (massive-neutrino fluid approximation switched off);
# tolerances and k sampling are tighter than the defaults but moderated so that
# one run takes a few minutes on one CPU.
# Note: a fluid/streaming trigger k*tau must stay below the corresponding l_max,
# otherwise the truncated hierarchy is integrated where it reflects power.
REFERENCE = {
    "tol_perturbations_integration": 1e-7,
    "k_per_decade_for_pk": 30,
    "k_per_decade_for_bao": 200,
    "perturbations_sampling_stepsize": 0.02,
    "l_max_g": 50, "l_max_pol_g": 25, "l_max_ur": 150, "l_max_ncdm": 50,
    "tight_coupling_trigger_tau_c_over_tau_h": 0.005,
    "tight_coupling_trigger_tau_c_over_tau_k": 0.008,
    "radiation_streaming_trigger_tau_over_tau_k": 240,
    "radiation_streaming_trigger_tau_c_over_tau": 100,
    "ur_fluid_trigger_tau_over_tau_k": 50,
    "ncdm_fluid_approximation": 3,          # 3 = none (no fluid approximation)
    "tol_ncdm_synchronous": 1e-6,
}

# Even tighter hierarchy/switch settings, only to check that REFERENCE is stable
REFERENCE_TIGHTER = {
    "l_max_g": 70, "l_max_pol_g": 40, "l_max_ur": 200, "l_max_ncdm": 70,
    "radiation_streaming_trigger_tau_over_tau_k": 500,
    "ur_fluid_trigger_tau_over_tau_k": 100,
    "tol_ncdm_synchronous": 1e-8,
}

# Numerical controls varied one at a time (the others kept at REFERENCE).
# Each one has the CLASS default value and a looser value.
# (CLASS refuses equal ur and ncdm fluid triggers, hence 10 and 11.)
CONTROLS = {
    "ODE tolerance": {
        "default": {"tol_perturbations_integration": 1e-5},
        "loose":   {"tol_perturbations_integration": 1e-3}},
    "k sampling": {
        "default": {"k_per_decade_for_pk": 10, "k_per_decade_for_bao": 70},
        "loose":   {"k_per_decade_for_pk": 5,  "k_per_decade_for_bao": 20}},
    "time sampling": {
        "default": {"perturbations_sampling_stepsize": 0.1},
        "loose":   {"perturbations_sampling_stepsize": 0.3}},
    "hierarchy l_max": {
        "default": {"l_max_g": 12, "l_max_pol_g": 10, "l_max_ur": 17, "l_max_ncdm": 17},
        "loose":   {"l_max_g": 6,  "l_max_pol_g": 6,  "l_max_ur": 8,  "l_max_ncdm": 8}},
    "tight-coupling switch": {
        "default": {"tight_coupling_trigger_tau_c_over_tau_h": 0.015,
                    "tight_coupling_trigger_tau_c_over_tau_k": 0.01},
        "loose":   {"tight_coupling_trigger_tau_c_over_tau_h": 0.05,
                    "tight_coupling_trigger_tau_c_over_tau_k": 0.03}},
    "late-radiation switches": {
        "default": {"radiation_streaming_trigger_tau_over_tau_k": 45,
                    "radiation_streaming_trigger_tau_c_over_tau": 5,
                    "ur_fluid_trigger_tau_over_tau_k": 30,
                    "ncdm_fluid_approximation": 2,      # 2 = CLASS default scheme
                    "ncdm_fluid_trigger_tau_over_tau_k": 31},
        "loose":   {"radiation_streaming_trigger_tau_over_tau_k": 15,
                    "radiation_streaming_trigger_tau_c_over_tau": 2,
                    "ur_fluid_trigger_tau_over_tau_k": 10,
                    "ncdm_fluid_approximation": 2,
                    "ncdm_fluid_trigger_tau_over_tau_k": 11}},
    "ncdm momentum sampling": {
        "default": {"tol_ncdm_synchronous": 1e-3},
        "loose":   {"tol_ncdm_synchronous": 1e-2}},
}


def interp_pk(k_from, P_from, k_to):
    """Cubic spline of ln P in ln k (used to put runs with different k grids on one grid)."""
    spline = CubicSpline(np.log(k_from), np.log(P_from))
    return np.exp(spline(np.log(k_to)))


# ---------------------------------------------------------------------------
# (c) CAMB with parameters matched to the CLASS run
# ---------------------------------------------------------------------------

def camb_params(class_params, omega_nu, YHe, accuracy_boost=1, k_per_logint=0,
                accurate_nu=False, match_omega_nu=True):
    """
    CAMB parameters matched to a CLASS parameter dict.

    Matched choices:
      - h, omega_b, omega_cdm, A_s, n_s, k_pivot, tau_reio, T_cmb = 2.7255 K (CLASS default)
      - one massive neutrino, N_eff = 3.044; the mass passed to CAMB is rescaled so that
        CAMB's omnuh2 equals CLASS's omega_nu (the two codes convert m_nu -> omega_nu differently)
      - Y_He fixed to the value CLASS used (BBN)
      - tanh reionization with the CLASS widths (Delta z = 0.5 for H and for He)
      - recombination: CAMB's default Recfast (compare with CLASS 'recombination = RECFAST')
      - linear spectrum only, synchronous gauge (CDM frame)
    match_omega_nu=False passes m_nu directly (naive, unmatched neutrino definition).
    """
    h = float(class_params["h"])
    m_nu = float(class_params["m_ncdm"])

    def make(mnu):
        p = camb.CAMBparams()
        p.set_cosmology(H0=100 * h, ombh2=float(class_params["omega_b"]),
                        omch2=float(class_params["omega_cdm"]),
                        mnu=mnu, num_massive_neutrinos=1, nnu=3.044,
                        tau=float(class_params["tau_reio"]), TCMB=2.7255, YHe=YHe)
        p.InitPower.set_params(As=float(class_params["A_s"]), ns=float(class_params["n_s"]),
                               pivot_scalar=float(class_params["k_pivot"]))
        p.Reion.delta_redshift = 0.5             # CLASS reionization_width
        p.Reion.helium_delta_redshift = 0.5      # CLASS helium_fullreio_width
        p.set_accuracy(AccuracyBoost=accuracy_boost)
        p.Transfer.accurate_massive_neutrinos = accurate_nu
        p.set_matter_power(redshifts=[float(class_params["z_pk"])], kmax=20.0,
                           k_per_logint=k_per_logint, nonlinear=False)
        p.NonLinear = camb.model.NonLinear_none
        return p

    p = make(m_nu)
    if match_omega_nu:
        # omnuh2 is proportional to m_nu, so one rescaling matches it exactly
        p = make(m_nu * omega_nu / p.omnuh2)
    return p


def camb_linear_pk(pars, k):
    """Linear total-matter P(k) from CAMB (var 'delta_tot'), k in 1/Mpc, P in Mpc^3."""
    results = camb.get_results(pars)
    pk = results.get_matter_power_interpolator(nonlinear=False, hubble_units=False,
                                               k_hunit=False, var1="delta_tot",
                                               var2="delta_tot")
    z = pars.Transfer.PK_redshifts[0]
    return pk.P(z, k), results
