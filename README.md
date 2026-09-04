# Numerical Optimization for Large-Scale Problems

Unconstrained optimization with derivative-based methods: **Modified Newton**
and **Truncated Newton**, applied to two large-scale test problems.

Project for the *Numerical Optimization* course — Laurea Magistrale in
Ingegneria Matematica, Politecnico di Torino, A.Y. 2025–2026.

**Authors:** Francesca Bagnato, Gianmarco Foni, Giovanna Maccarone

---

## Overview

This repository implements and compares two derivative-based unconstrained
optimization methods on large-scale problems, using both exact and
finite-difference approximated derivatives:

- **Modified Newton method** — standard version and an eigenvalue-flipping
  variant for Hessian correction when it is not positive definite.
- **Truncated Newton method** — standard version and a Hessian-free variant
  based on the conjugate gradient method.

The methods are tested on two problems from More–Garbow–Hillstrom's
classical test set:

- **Problem 16 — Banded Trigonometric** (`Trig16`)
- **Problem 31 — Broyden Tridiagonal least-squares** (`Broyden31`)

for increasing problem dimensions (n = 2, 10³, 10⁴, 10⁵) and multiple
random starting points, in order to study convergence behavior,
convergence rate, and scalability.

## Finite Differences

Besides exact analytical derivatives, the repository includes a
finite-difference implementation of gradients and Hessians, tested under
two settings:

- **Case 1** — exact gradient, finite-difference Hessian
- **Case 2** — both gradient and Hessian approximated via finite differences

for different finite-difference step sizes (controlled by a parameter `k`),
in order to assess the impact of derivative approximation error on
convergence.

## Repository structure

> Adjust this section to match your actual folder layout.

```
.
├── solvers/                 # Method implementations (Modified Newton, Truncated Newton, variants)
├── problems/                # Test problem definitions (Trig16, Broyden31)
├── fd/                      # Scripts for the finite differences
├── plot/                    # Functions for plot and tables
├── utils/                   # Scripts for diagnostic and tuning
├── Documentation/           # PDF report
└── README.md
```

## Requirements

- MATLAB (or Octave) 
- No additional toolboxes beyond base MATLAB, unless otherwise noted

## How to run

```matlab
% Example — adjust to your actual entry point
run('experiments/run_trig16.m')
run('experiments/run_broyden31.m')
```

Results (convergence rates, iteration counts, stopping flags) are saved to
the `graphs_trig16/` and `graphs_broyden31/`  folder and summarized in tables and plots included in the
report.

## Key results (summary)

- **Trig16**: with exact derivatives, Modified Newton achieves rates close
  to quadratic (≈2.3–2.5), Truncated Newton around 1.5. Both methods
  deteriorate sharply at n = 10⁵ due to the ill-conditioning of the
  problem's Hessian. Finite-difference derivatives (especially Case 2)
  further reduce robustness.
- **Broyden31**: both methods converge from all starting points and all
  dimensions with exact derivatives. Convergence rates are stable across n.
  With finite differences, Case 1 remains essentially unaffected, while
  Case 2 breaks down as the FD step size decreases (all runs fail at
  k = 12 for n ≥ 10³, hitting the maximum iteration limit).

Full numerical results, tables, and figures are available in the project
report (`report/`).

## License

Specify a license (e.g., MIT) or state that this is coursework not intended
for reuse without permission.
