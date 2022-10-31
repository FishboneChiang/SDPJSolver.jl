# SDPSolver
A native Julia semidefinite program (SDP) solver.
- Motivated by the eft/modular bootstrap programs, SDPSolver is a parallelized, arbitrary precision SDP solver that uses the primal-dual interior-point method. 
- SDPSolver is largely inspired by SDPA and SDPB, with slightly different parallelization architecture.
- The solver is still in a development stage, which is far from fully optimized and might contain bugs. Corrections and suggestions are welcome and will get serious attention : )

The function
> sdp(prec, c, A, C, B, b, β, Ωp, Ωq, ϵ_gap, ϵ_primal, ϵ_dual, iterMax)

solves the following SDP:

## Primal problem
    Minimize    c^T x
    subject to  X = ∑ x_i A_i - C
                B^T x = b
                X >= 0

## Dual problem
    Maximize    tr(C Y) + b^T y 
    subject to  tr(A_* Y) + B y + c = 0
                Y >= 0 

## Domain
    x ∈ R^m
    X, Y ∈ S^k
    A_i, C ∈ S^k 
    B ∈ R^(m*n)
    b, y ∈ R^n
    
## Interior point method
In each iteration, the program solves the following deformed KKT conditions using Newton's method:
- Primal feasibility, 
- Dual feasibility, 
- Complementarity: tr(XY) = μI.
The Mehrotra predictor-corrector trick is implemented.

## Arguments

`prec`: arithemetic precision in base-10, which is equivalent to
```jldoctest
setprecision(prec, base = 10)
```

`β`: factor of reduction in μ in each step

`Ωp` and `Ωd` are initial values for the matrices X and Y: $X = Ω_q I, Y = Ω_d I$.

The iteration terminates if
- the duality gap, primal infeasibility, and dual infeasibility are below `ϵ_gap`, `ϵ_primal`, and `ϵ_dual`, respectively, or
- the number of iteration exceeds `iterMax`.
