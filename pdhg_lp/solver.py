
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse

@dataclass(frozen=True)
class PDHGResult:
    x: np.ndarray
    y: np.ndarray
    objective: float
    primal_residual: float
    projected_dual_residual: float
    iterations: int
    status: str
    history: tuple

def estimate_spectral_norm(A,iterations=40,seed=0):
    A=sparse.csr_matrix(A); rng=np.random.default_rng(seed)
    x=rng.normal(size=A.shape[1]); x/=max(np.linalg.norm(x),1e-12)
    for _ in range(iterations):
        y=A@x
        z=A.T@y
        nz=np.linalg.norm(z)
        if nz<1e-15:return 0.0
        x=z/nz
    return float(np.linalg.norm(A@x))

def projected_dual_residual(c,A,y,x,lo,hi):
    # KKT fixed-point residual for box-constrained primal stationarity.
    g=np.asarray(c)+A.T@np.asarray(y)
    projected=np.clip(np.asarray(x)-g,lo,hi)
    return float(np.linalg.norm(np.asarray(x)-projected)/(1+np.linalg.norm(x)))

def solve_pdhg(problem,max_iter=20000,tol=1e-6,check_every=100,theta=1.0):
    c=np.asarray(problem.c,float); A=sparse.csr_matrix(problem.A)
    b=np.asarray(problem.b,float); lo=np.asarray(problem.lower,float); hi=np.asarray(problem.upper,float)
    normA=estimate_spectral_norm(A)
    if normA<=1e-15:
        raise ValueError("zero operator not supported")
    tau=.70/normA; sigma=.70/normA
    x=np.clip(np.zeros_like(c),lo,hi); xbar=x.copy(); y=np.zeros_like(b)
    xsum=np.zeros_like(x); ysum=np.zeros_like(y)
    hist=[]
    status="MAX_ITER"
    for k in range(1,max_iter+1):
        ynew=y+sigma*(A@xbar-b)
        xnew=np.clip(x-tau*(c+A.T@ynew),lo,hi)
        xbar=xnew+theta*(xnew-x)
        x,y=xnew,ynew
        xsum += x
        ysum += y
        if k%check_every==0 or k==1 or k==max_iter:
            pr=float(np.linalg.norm(A@x-b)/(1+np.linalg.norm(b)))
            dr=projected_dual_residual(c,A,y,x,lo,hi)
            obj=float(c@x)
            hist.append((k,obj,pr,dr))
            if max(pr,dr)<=tol:
                status="CONVERGED"; break
    # First-order saddle iterations can oscillate even when the ergodic
    # sequence is already accurate. Return whichever of current/ergodic
    # iterates has the smaller combined KKT residual.
    xa=xsum/max(k,1); ya=ysum/max(k,1)
    pra=float(np.linalg.norm(A@xa-b)/(1+np.linalg.norm(b)))
    dra=projected_dual_residual(c,A,ya,xa,lo,hi)
    if max(pra,dra) < max(pr,dr):
        x,y,pr,dr=xa,ya,pra,dra
    return PDHGResult(x,y,float(c@x),pr,dr,k,status,tuple(hist))
