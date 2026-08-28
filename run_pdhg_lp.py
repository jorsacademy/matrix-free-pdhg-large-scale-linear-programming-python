
from __future__ import annotations
import argparse, time, numpy as np
from pdhg_lp import generate_sparse_feasible_lp,solve_pdhg,solve_highs

def self_test():
    p=generate_sparse_feasible_lp(seed=1,n=30,m=8,density=.2)
    r=solve_pdhg(p,max_iter=10000,tol=1e-5,check_every=50)
    xh,oh=solve_highs(p)
    gap=abs(r.objective-oh)/max(abs(oh),1.0)
    assert r.primal_residual<1e-4 and gap<2e-3
    print("Matrix-free PDHG self-test: OK")

def main(a):
    p=generate_sparse_feasible_lp(seed=a.seed,n=a.n,m=a.m,density=a.density)
    t=time.perf_counter(); r=solve_pdhg(p,max_iter=a.max_iter,tol=a.tol,check_every=a.check_every); tp=time.perf_counter()-t
    print(f"shape={p.A.shape} nnz={p.A.nnz} density={p.A.nnz/(p.A.shape[0]*p.A.shape[1]):.6f}")
    print(f"PDHG status={r.status} iterations={r.iterations} time={tp:.3f}s obj={r.objective:.6f} primal_res={r.primal_residual:.3e} projected_dual_res={r.projected_dual_residual:.3e}")
    if a.highs:
        t=time.perf_counter(); _,oh=solve_highs(p); th=time.perf_counter()-t
        rel=abs(r.objective-oh)/max(abs(oh),1.0)
        print(f"HiGHS obj={oh:.6f} time={th:.3f}s relative objective error={rel:.3e}")

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--seed",type=int,default=42); p.add_argument("--n",type=int,default=1000); p.add_argument("--m",type=int,default=250)
    p.add_argument("--density",type=float,default=.01); p.add_argument("--max-iter",type=int,default=12000)
    p.add_argument("--tol",type=float,default=1e-5); p.add_argument("--check-every",type=int,default=100); p.add_argument("--highs",action="store_true")
    return p.parse_args()
if __name__=="__main__":
    a=parse(); self_test() if a.self_test else main(a)
