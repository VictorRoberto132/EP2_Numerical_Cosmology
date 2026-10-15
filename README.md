# EP2 - Numerical Cosmology

Solutions of the second exercise sheet (EP2, Lectures 5-9) of the PhD course *Numerical Cosmology*, Institute of Physics, University of São Paulo (IFUSP).

**Author:** Victor Roberto Soares da Silva

## Layout

| Folder | Content |
|---|---|
| `Library/` | Python code, one module per exercise (`exercise_N.py`) |
| `notebooks/` | One notebook per exercise (`exercise_N.ipynb`), which runs the items and makes the figures/tables |
| `results/exercise_N/` | Saved data (`.csv`, `.npz`), figures (`fig_*.pdf`) and LaTeX tables (`tab_*.tex`) |
| `report/` | LaTeX report of Exercise 4 (`.tex` and `.pdf`) |

## What was done

1. **Linear growth (Lecture 5).** Growth ODE in N = ln a for EdS, flat ΛCDM and open CDM, returning D(a) and f(a). It is checked against the integral solution (quadrature), and we test how much the result depends on the starting epoch a_i and the solver tolerances.
2. **Mode coupling (Lecture 6).** Second-order SPT density (F2 kernel, computed with FFTs and zero padding), with power spectra and bispectra of δ1 and δ1 + δ2. The ensemble mean is compared with B_tree, plus tests of resolution, box size and amplitude. Supplementary: the one-loop P22 + 2P13.
3. **Lagrangian initial conditions (Lecture 7).** Gaussian field in a periodic box, then Zel'dovich positions and velocities. Each step is validated, and the input spectrum is recovered from the particles (CIC).
4. **Spherical collapse (Lecture 8).** Three independent formulations (A: δ, B: δ-ϑ, C: g = 1/δ), solved by shooting for the collapse epoch. Outputs: δc, turnaround and virial overdensities. Checks: EdS benchmarks, the Ωm0 × zc grid, and convergence. Includes the LaTeX report.
5. **Boltzmann solver (Lecture 9).** CLASS v3.4.0 with the fiducial model of the notes (eqs. 8.1-8.2), linear P_m at z = 0:
   - (a) reconstruct P_m from the transfer T_m and compare with the direct output (identity check, residual ~1e-12);
   - (b) convergence: tighter reference run, then one numerical control at a time (ODE tolerance, k and time sampling, ℓ_max, approximation switches, neutrino momentum sampling);
   - (c) CLASS vs CAMB with matched parameters, keeping physical-model changes (recombination code, neutrino-mass convention) separate from precision changes.

   The lecture's `class_base.ini` is not available here, so it was rebuilt in `Library/exercise_5_class_base.ini`. The supplementary checks of Exercise 5 were not done.

## Requirements

- Python 3.11 with `numpy`, `scipy`, `matplotlib`, `pandas`, `jinja2` (LaTeX tables)
- Jupyter with `ipympl` (the notebooks use `%matplotlib widget`)
- `camb` (Exercise 5; tested with 2.0.5)
- CLASS v3.4.0 **executable** (Exercise 5), built from source:
  ```bash
  git clone https://github.com/lesgourg/class_public.git
  cd class_public && git checkout v3.4.0 && make class
  ```
  Set its path in `CLASS_EXE` at the top of `Library/exercise_5.py`, or export the environment variable `CLASS_EXE=/path/to/class_public/class`.
- `pdflatex` / `latexmk` (only to compile the report of Exercise 4)

```bash
pip install numpy scipy matplotlib pandas jinja2 jupyter ipympl camb
```

## How to run

Each notebook writes its outputs to `results/exercise_N/`. Run them from the `notebooks/` folder (they import `../Library`), either interactively in Jupyter or with:

```bash
cd notebooks
jupyter execute --inplace exercise_1.ipynb
jupyter execute --inplace exercise_2.ipynb             # ~7 min
jupyter execute --inplace exercise_3.ipynb
jupyter execute --inplace exercise_4.ipynb             # ~1 min
jupyter execute --inplace --timeout=-1 exercise_5.ipynb
```

Exercise 4 report (reads the figures and tables from `results/exercise_4/`):

```bash
cd report
latexmk -pdf exercise_4.tex
```

**Exercise 5 runtime.** The input, log and output tables of every CLASS run are saved in `results/exercise_5/class_runs/`, and existing runs are reused. With these files the notebook runs in under a minute. Without them, all CLASS runs are recomputed, which takes about 1 hour on one CPU (the reference runs use full neutrino hierarchies).
