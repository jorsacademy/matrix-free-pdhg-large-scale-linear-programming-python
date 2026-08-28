
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

def solve_highs(problem):
    result=linprog(problem.c,A_eq=problem.A,b_eq=problem.b,
                   bounds=list(zip(problem.lower,problem.upper)),method="highs")
    if not result.success: raise RuntimeError(result.message)
    return np.asarray(result.x),float(result.fun)
