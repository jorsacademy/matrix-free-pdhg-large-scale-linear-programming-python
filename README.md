# Matrix-Free PDHG for Large-Scale Linear Programming

An educational first-order LP solver for box-constrained equality-form linear programs:

```text
minimize    c^T x
subject to  A x = b
            lower <= x <= upper
```

The implementation is deliberately **matrix-free with respect to linear solves**: every iteration requires only sparse matrix-vector products `A @ x` and `A.T @ y`, vector arithmetic and box projection. There is no KKT factorization.

## PDHG iteration

Using the saddle formulation:

```text
min_x max_y c^T x + y^T(Ax-b) + I_[l,u](x)
```

the solver applies:

```text
y_{k+1} = y_k + sigma * (A xbar_k - b)

x_{k+1} =
projection_[l,u](
    x_k - tau * (c + A^T y_{k+1})
)

xbar_{k+1} =
x_{k+1} + theta * (x_{k+1} - x_k)
```

The spectral norm of `A` is estimated by sparse power iteration and a conservative step size satisfies the standard PDHG stability condition.

An ergodic average is tracked because first-order saddle iterates may oscillate even when averaged feasibility is accurate.

## Diagnostics

The code reports:

- primal equality residual;
- projected primal-stationarity/KKT residual;
- objective;
- iteration count;
- convergence status.

`MAX_ITER` is not relabeled as convergence merely because the objective is close to an oracle.

## HiGHS oracle

Small and medium fixtures can be solved independently with SciPy/HiGHS. Regression tests require PDHG feasibility and objective agreement on several conditioned sparse LPs.

## Sparse benchmark generator

The synthetic generator creates a full-row-rank sparse equality operator with an identity anchor plus a sparse random block. This avoids turning the benchmark into a test of accidental near-singularity while retaining sparse matrix-vector behavior.

The generator also constructs a known feasible point and sets:

```text
b = A x_feasible
```

## Development run

```text
rows       180
columns    800
nnz       1854
density   1.2875%
```

At 8,000 iterations:

```text
PDHG status                    MAX_ITER
PDHG objective                -1226.907921
primal residual                  7.349e-05
projected dual residual          1.480e-04

HiGHS objective               -1226.911882
relative objective error          3.229e-06
```

PDHG had not met the requested `1e-5` residual tolerance, so the run is correctly reported as `MAX_ITER`, despite a very small objective difference.

No speedup claim is made: on this moderate CPU fixture HiGHS is much faster. The point is to demonstrate the algorithmic structure that becomes attractive when factorization memory is the bottleneck and matrix-vector operations can be parallelized.

## Tests

- spectral norm hand check;
- synthetic feasibility;
- PDHG vs HiGHS on several small LPs;
- box-bound preservation;
- sparse operator retention.

## Run

```bash
pip install -r requirements.txt
python run_pdhg_lp.py --self-test
python -m unittest discover -s tests -v
python run_pdhg_lp.py --highs
```

## Scope

Not claimed:

- this is Google PDLP or a reproduction of its production enhancements;
- PDHG beats simplex/barrier on the supplied CPU fixture;
- `MAX_ITER` means solved;
- the simple generator represents industrial LP conditioning;
- the implementation contains advanced presolve, scaling, adaptive restart or GPU kernels.
