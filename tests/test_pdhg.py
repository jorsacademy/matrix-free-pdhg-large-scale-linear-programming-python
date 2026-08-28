
import unittest, numpy as np
from scipy import sparse
from pdhg_lp import *

class Tests(unittest.TestCase):
    def test_spectral_norm_diagonal(self):
        A=sparse.diags([1.,2.,5.])
        self.assertAlmostEqual(estimate_spectral_norm(A,80),5.,places=5)

    def test_generator_feasible(self):
        p=generate_sparse_feasible_lp(seed=2,n=60,m=15,density=.1)
        self.assertLess(np.linalg.norm(p.A@p.known_feasible-p.b),1e-9)

    def test_pdhg_matches_highs_small(self):
        for seed in [3,4,5]:
            p=generate_sparse_feasible_lp(seed=seed,n=40,m=10,density=.2)
            r=solve_pdhg(p,max_iter=25000,tol=2e-6,check_every=50)
            _,obj=solve_highs(p)
            self.assertLess(r.primal_residual,2e-5)
            self.assertLess(abs(r.objective-obj)/max(abs(obj),1),3e-3)

    def test_box_bounds_respected(self):
        p=generate_sparse_feasible_lp(seed=6,n=50,m=12,density=.15)
        r=solve_pdhg(p,max_iter=5000,tol=1e-4,check_every=50)
        self.assertTrue(np.all(r.x>=p.lower-1e-12))
        self.assertTrue(np.all(r.x<=p.upper+1e-12))

    def test_sparse_operator_retained(self):
        p=generate_sparse_feasible_lp(seed=7,n=200,m=50,density=.03)
        self.assertTrue(sparse.isspmatrix_csr(p.A))
        self.assertLess(p.A.nnz,p.A.shape[0]*p.A.shape[1]//5)

if __name__=="__main__": unittest.main()
