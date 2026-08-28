
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse

@dataclass(frozen=True)
class EqualityBoxLP:
    c: np.ndarray
    A: sparse.csr_matrix
    b: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    known_feasible: np.ndarray | None = None

    def __post_init__(self):
        c=np.asarray(self.c,float); b=np.asarray(self.b,float)
        lo=np.asarray(self.lower,float); hi=np.asarray(self.upper,float)
        A=sparse.csr_matrix(self.A)
        if A.shape!=(len(b),len(c)) or lo.shape!=c.shape or hi.shape!=c.shape:
            raise ValueError("shape mismatch")
        if np.any(lo>hi): raise ValueError("lower > upper")

def generate_sparse_feasible_lp(*,seed=42,n=200,m=60,density=.04):
    if n <= m:
        raise ValueError("generator requires n > m")
    rng=np.random.default_rng(seed)

    # Identity anchor guarantees full row rank and avoids pathological
    # equality-conditioning artifacts while the remaining block stays sparse.
    rest=sparse.random(
        m,n-m,density=density,format="csr",random_state=rng,
        data_rvs=lambda k:rng.normal(scale=.6,size=k)
    )
    A=sparse.hstack([sparse.eye(m,format="csr"),rest],format="csr")

    lo=np.zeros(n); hi=rng.uniform(2,8,n)
    x0=rng.uniform(.15,.85,n)*hi
    b=A@x0
    c=rng.normal(0,1,n)
    return EqualityBoxLP(c,A,b,lo,hi,x0)

