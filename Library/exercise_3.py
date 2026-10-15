"""
Exercise 3 - Lagrangian initial conditions.

Periodic cube of side L with an Ng^3 mesh. Reference: Lecture 7, notes
Sections 7.1-7.6 and Appendix B.

Fourier convention (Lecture 7, eqs. 7.5-7.7):
    k = (2 pi / L) n,   n integer vector
    delta(x) = (1/V) sum_k delta_k exp(i k.x)
    delta_k  = int_V d^3x delta(x) exp(-i k.x)        [delta_k has units of length^3]
"""

import numpy as np


def power_law_spectrum(k, A, n):
    """
    Input linear spectrum P_L(k, a_ref) = A k^n, with T(k) = 1
    (Lecture 5, eq. 4.36). Units: if k is in h/Mpc, P is in (Mpc/h)^3.
    """
    return A * k**n


def wavevectors(L, Ng):
    """
    Wavevector components k_x, k_y, k_z on the mesh, in numpy's FFT storage order.

    np.fft.fftfreq(Ng) * Ng gives the integer index n = 0, 1, ..., Ng/2 - 1, -Ng/2, ..., -1.
    Note: for even Ng the Nyquist index is stored as -Ng/2 (it is the same cell as +Ng/2).
    """
    n = np.fft.fftfreq(Ng) * Ng          # integer Fourier indices along one axis
    k1d = 2.0 * np.pi / L * n            # k = (2 pi / L) n  (eq. 7.5)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    return kx, ky, kz


def gaussian_field_k(L, Ng, A, n, seed):
    """
    Draw the Gaussian Fourier coefficients delta_ref,k with <|delta_k|^2> = V P_L(k)
    (Lecture 7, Sections 7.5-7.6).

    - Non-self-conjugate pair: draw one member as sqrt(V P / 2) (g1 + i g2)  (eq. 7.15)
      and set its partner delta_{-k} = conj(delta_k)                       (eq. 7.17).
    - Self-conjugate cell (n = -n mod Ng): real, sqrt(V P) g                (eq. 7.20).
    - Zero mode: delta_0 = 0                                                (eq. 7.21).

    Returns delta_k (complex array, units of length^3) in numpy's FFT storage order.
    """
    V = L**3
    rng = np.random.default_rng(seed)

    # |k| on the mesh, and the spectrum at every nonzero k
    kx, ky, kz = wavevectors(L, Ng)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    P = np.zeros_like(k)
    P[k > 0] = power_law_spectrum(k[k > 0], A, n)

    # Index of the partner cell -n (mod Ng) along one axis
    idx = np.arange(Ng)
    neg = (-idx) % Ng

    # Flat label of each cell and of its partner, used to pick one member of each pair:
    # the cell with the smaller label is the one we draw ("independent"),
    # the cell with the larger label gets the complex conjugate.
    i, j, l = np.meshgrid(idx, idx, idx, indexing="ij")
    label = (i * Ng + j) * Ng + l
    label_partner = (neg[i] * Ng + neg[j]) * Ng + neg[l]
    independent = label < label_partner
    self_conjugate = label == label_partner

    delta_k = np.zeros((Ng, Ng, Ng), dtype=complex)

    # Independent member of each pair: eq. 7.15
    g1 = rng.standard_normal(independent.sum())
    g2 = rng.standard_normal(independent.sum())
    delta_k[independent] = np.sqrt(V * P[independent] / 2.0) * (g1 + 1j * g2)

    # Self-conjugate cells (zero mode and Nyquist corners/edges/faces): real, eq. 7.20
    g = rng.standard_normal(self_conjugate.sum())
    delta_k[self_conjugate] = np.sqrt(V * P[self_conjugate]) * g

    # Partner cells: delta_{-k} = conj(delta_k), eq. 7.17.
    # delta_k[np.ix_(neg, neg, neg)] is the array of delta_{-k} at each k.
    delta_minus_k = delta_k[np.ix_(neg, neg, neg)]
    partner = label > label_partner
    delta_k[partner] = np.conj(delta_minus_k[partner])

    # Zero mode: eq. 7.21
    delta_k[0, 0, 0] = 0.0

    return delta_k


def field_x(delta_k, L):
    """
    Real-space field delta(x_j) = (1/V) sum_k delta_k exp(i k.x_j).

    numpy's ifftn computes (1/Ng^3) sum_n c_n exp(2 pi i n.j / Ng), and k.x_j = 2 pi n.j / Ng,
    so delta(x_j) = (Ng^3 / V) * ifftn(delta_k)   (equivalently eq. 7.13: c_n = Ng^3 delta_k / V).
    The imaginary part is only round-off, because delta_k is Hermitian.
    """
    Ng = delta_k.shape[0]
    V = L**3
    return (Ng**3 / V * np.fft.ifftn(delta_k)).real


def power_spectrum_from_k(delta_k, L, nbins):
    """
    Shell-averaged power spectrum of the drawn Fourier modes:
        P_hat(k_bin) = (1/V) <|delta_k|^2>_shell,
    i.e. the estimator of <|delta_k|^2> = V P_L(k) (Lecture 7, eq. 7.9).
    Bins are linear shells of width k_f = 2 pi / L, from k_f/2; the zero mode is excluded.

    Returns the mean |k| in each bin, P_hat, and the number of modes per bin.
    """
    V = L**3
    k_f = 2.0 * np.pi / L

    # |k| of every cell and |delta_k|^2 / V
    kx, ky, kz = wavevectors(L, Ng=delta_k.shape[0])
    k = np.sqrt(kx**2 + ky**2 + kz**2).ravel()
    power = (np.abs(delta_k)**2 / V).ravel()

    # Shell edges: k_f/2, 3 k_f/2, ...  (each shell centred on an integer multiple of k_f)
    edges = k_f * (np.arange(nbins + 1) + 0.5)
    which = np.digitize(k, edges) - 1        # bin index of each cell (-1 or nbins = outside)
    inside = (which >= 0) & (which < nbins)

    # Number of modes, mean |k| and mean power in each shell
    n_modes = np.bincount(which[inside], minlength=nbins)
    k_mean = np.bincount(which[inside], weights=k[inside], minlength=nbins) / n_modes
    P_hat = np.bincount(which[inside], weights=power[inside], minlength=nbins) / n_modes

    return k_mean, P_hat, n_modes


def field_k(delta_x, L):
    """
    Fourier coefficients delta_k = int_V d^3x delta(x) exp(-i k.x), inverse of field_x.

    numpy's fftn computes sum_j delta_j exp(-2 pi i n.j / Ng), and the integral is
    approximated by the sum times the cell volume (L/Ng)^3, so
    delta_k = (V / Ng^3) * fftn(delta_x)   (eq. 7.13).
    """
    Ng = delta_x.shape[0]
    V = L**3
    return V / Ng**3 * np.fft.fftn(delta_x)


def derivative_wavevectors(L, Ng):
    """
    Wavevector components used in odd derivatives (i k_i), with the Nyquist convention of
    Section 7.6: k_i = 0 on the Nyquist cell of axis i (n_i = -Ng/2). The same operator is used
    to build the displacement in (b) and to validate it in (c).
    """
    k_d = []
    for ki in wavevectors(L, Ng):
        ki_d = ki.copy()
        ki_d[np.isclose(np.abs(ki), np.pi * Ng / L)] = 0.0
        k_d.append(ki_d)
    return k_d


# ---------------------------------------------------------------------------
# (b) Zel'dovich approximation (Lecture 7, Section 7.8 and Appendix B)
# ---------------------------------------------------------------------------

def displacement_k(delta_k, L, D1):
    """
    ZA displacement in Fourier space (eq. 7.25):
        Psi_k = i D1 (k / k^2) delta_ref,k,   Psi_0 = 0 (eq. 7.21).

    Nyquist convention (Section 7.6): for this odd derivative (i k_i), the component k_i is set
    to zero on the Nyquist cell of axis i (n_i = -Ng/2), so that Psi stays real. The inverse
    Laplacian uses the same derivative wavevector k_d (Section 7.8: same discrete derivative and
    inverse Laplacian in construction and test), so that -div Psi = D1 delta_ref holds exactly:
        Psi_k = i D1 (k_d / k_d^2) delta_ref,k.
    Away from the Nyquist planes k_d = k, so this is eq. 7.25. Cells with k_d = 0 but k != 0
    (self-conjugate cells whose components are all 0 or Nyquist) get Psi_k = 0.

    Returns [Psi_x,k, Psi_y,k, Psi_z,k], each a complex Ng^3 array (units of length^4).
    """
    Ng = delta_k.shape[0]
    k_d = derivative_wavevectors(L, Ng)
    kd2 = k_d[0]**2 + k_d[1]**2 + k_d[2]**2

    # 1/k_d^2, set to zero where k_d = 0 (includes the zero mode, Psi_0 = 0)
    inv_kd2 = np.zeros_like(kd2)
    inv_kd2[kd2 > 0] = 1.0 / kd2[kd2 > 0]

    # i k_d / k_d^2 for each component
    Psi_k = []
    for ki_d in k_d:
        Psi_k.append(1j * D1 * ki_d * inv_kd2 * delta_k)
    return Psi_k


def lattice(L, Np):
    """
    Regular particle load q_j = (L / Np) j, j_i = 0, ..., Np - 1 (eq. 7.23 with offset 0).
    With Np = Ng the labels coincide with the mesh nodes, so no interpolation is needed.

    Returns q with shape (Np^3, 3), in the same (i, j, l) order as a flattened Ng^3 mesh.
    """
    q1d = L / Np * np.arange(Np)
    qx, qy, qz = np.meshgrid(q1d, q1d, q1d, indexing="ij")
    return np.column_stack([qx.ravel(), qy.ravel(), qz.ravel()])


def za_particles(delta_k, L, D1, f1, calH):
    """
    ZA positions and velocities at a_i (Section 7.8, steps 1-6), one particle per mesh node.

        x_p = (q_p + Psi_p) mod L       (eq. 7.26)
        v_p = calH_i f1_i Psi_p         (eq. 7.27),  calH = a H,  v = a dx/dt

    D1, f1, calH are evaluated at a_i. delta_k is delta_ref,k (spectrum at a_ref),
    so the growth D1(a_i) is applied only here, once.

    Returns q, x, v, Psi, each with shape (Ng^3, 3).
    """
    Ng = delta_k.shape[0]

    # Displacement on the mesh: inverse transform each component (real up to round-off)
    Psi_k = displacement_k(delta_k, L, D1)
    Psi = np.column_stack([field_x(P, L).ravel() for P in Psi_k])

    # Mean displacement set to zero (eq. 8.7); already ~0 since Psi_0 = 0
    Psi -= Psi.mean(axis=0)

    # Particle labels at the mesh nodes, displaced and wrapped into [0, L)
    q = lattice(L, Ng)
    x = np.mod(q + Psi, L)

    # Velocities
    v = calH * f1 * Psi
    return q, x, v, Psi


# ---------------------------------------------------------------------------
# Mass assignment (Lecture 7, Section 8.2)
# ---------------------------------------------------------------------------

def cic_mass(x, L, Ng):
    """
    Cloud-in-cell assignment of unit-mass particles onto an Ng^3 periodic mesh with nodes at
    (L/Ng) j. Returns the mass in each cell (its sum is the number of particles).
    """
    dx = L / Ng
    s = x / dx                        # positions in units of cells
    i0 = np.floor(s).astype(int)      # lower node along each axis
    w1 = s - i0                       # weight of the upper node
    w0 = 1.0 - w1                     # weight of the lower node

    rho = np.zeros((Ng, Ng, Ng))
    # Loop over the 8 neighbouring nodes
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                w = ((w1[:, 0] if ox else w0[:, 0]) *
                     (w1[:, 1] if oy else w0[:, 1]) *
                     (w1[:, 2] if oz else w0[:, 2]))
                ix = (i0[:, 0] + ox) % Ng
                iy = (i0[:, 1] + oy) % Ng
                iz = (i0[:, 2] + oz) % Ng
                np.add.at(rho, (ix, iy, iz), w)

    return rho


def cic_density(x, L, Ng):
    """Density contrast delta_j = rho_j / rho_mean - 1 of the CIC mesh (eq. 8.9)."""
    rho = cic_mass(x, L, Ng)
    return rho / rho.mean() - 1.0



# ---------------------------------------------------------------------------
# (c) Validation (Lecture 7, Section 8.1)
# ---------------------------------------------------------------------------

def hermitian_residual(delta_k):
    """max |delta_k - conj(delta_{-k})| / max |delta_k|  (eq. 8.1); zero for an exact draw."""
    Ng = delta_k.shape[0]
    neg = (-np.arange(Ng)) % Ng
    delta_minus_k = delta_k[np.ix_(neg, neg, neg)]
    return np.abs(delta_k - np.conj(delta_minus_k)).max() / np.abs(delta_k).max()


def divergence_k(Psi_k, L):
    """Fourier divergence  i k_d . Psi_k, with the same derivative operator used in (b)."""
    Ng = Psi_k[0].shape[0]
    kx, ky, kz = derivative_wavevectors(L, Ng)
    return 1j * (kx * Psi_k[0] + ky * Psi_k[1] + kz * Psi_k[2])


def curl_k(Psi_k, L):
    """Fourier curl  i k_d x Psi_k, with the same derivative operator used in (b)."""
    Ng = Psi_k[0].shape[0]
    kx, ky, kz = derivative_wavevectors(L, Ng)
    return [1j * (ky * Psi_k[2] - kz * Psi_k[1]),
            1j * (kz * Psi_k[0] - kx * Psi_k[2]),
            1j * (kx * Psi_k[1] - ky * Psi_k[0])]


# ---------------------------------------------------------------------------
# (d) Spectrum recovery (Lecture 7, Sections 8.1.1, 8.2, 8.3)
# ---------------------------------------------------------------------------

def cic_window(L, Ng):
    """CIC window W(k) = prod_i [sin(k_i dx/2) / (k_i dx/2)]^2 (eq. 8.10)."""
    dx = L / Ng
    W = np.ones((Ng, Ng, Ng))
    for ki in wavevectors(L, Ng):
        W *= np.sinc(ki * dx / (2.0 * np.pi))**2   # np.sinc(y) = sin(pi y) / (pi y)
    return W


def measure_spectra(L, Ng, A, n, seed, D1, f1, calH, nbins):
    """
    Full pipeline for one realization: draw delta_ref (a), build ZA particles at a_i (b),
    assign them with CIC and measure the shell-averaged spectra (eq. 8.4).

    Returns k, P of the Gaussian field D1 delta_ref, P of the ZA particles (raw CIC),
    and P of the ZA particles with the CIC window divided out.
    """
    delta_k = gaussian_field_k(L, Ng, A, n, seed)
    q, x, v, Psi = za_particles(delta_k, L, D1, f1, calH)
    delta_za_k = field_k(cic_density(x, L, Ng), L)

    k, P_gauss, _ = power_spectrum_from_k(D1 * delta_k, L, nbins)
    k, P_za, _ = power_spectrum_from_k(delta_za_k, L, nbins)
    k, P_za_deconv, _ = power_spectrum_from_k(delta_za_k / cic_window(L, Ng), L, nbins)
    return k, P_gauss, P_za, P_za_deconv
