# SDPJSolver
SDPJ is a native Julia [semidefinite program](https://en.wikipedia.org/wiki/Semidefinite_programming) (SDP) solver.
- Motivated by the eft/modular bootstrap programs, SDPJ is a parallelized, arbitrary precision SDP solver based on the primal-dual interior-point method. 
- SDPJ is largely inspired by [SDPA](https://sdpa.sourceforge.net/) and [SDPB](https://github.com/davidsd/sdpb), with slightly different parallelization architecture.
- The solver is still in a development stage, which is far from fully optimized and might contain bugs. Corrections and suggestions are welcome and will get serious attention : )

## Installation
In the Julia REPL, hit `]` to enter the Pkg REPL, and then run
```julia
add https://github.com/FishboneChiang/SDPJSolver.jl
```

## The optimization problem
The function
```julia
sdp(c, A, C, B, b;
    β=0.1, γ=0.9, Ωp=1, Ωd=1,
    ϵ_gap=1e-10, ϵ_primal=1e-10, ϵ_dual=1e-10,
    iterMax=200, prec=300,
    restart=true, minStep=1e-10, maxOmega=1e50, OmegaStep=1e5)
```
solves the following SDP:

### Primal
$$
    \begin{aligned}
        \text{Minimize } \quad & c^T x \\
        \text{subject to } \quad & X^{(l)} = \sum_i x_i A_i^{(l)} - C^{(l)} \geq 0, \quad l = 1, 2, ..., L \\
        & B^T x = b 
    \end{aligned}
$$

### Dual
$$
    \begin{aligned}
        \text{Maximize } \quad & \sum_l tr(C^{(l)} Y^{(l)}) + b^T y \\
        \text{subject to } \quad & \sum_l tr(A_{\star}^{(l)} Y^{(l)}) + B y - c = 0 \\
        & Y^{(l)} \geq 0, \quad l = 1, 2, ..., L
    \end{aligned}
$$

### Domain
$$
    \begin{aligned}
        x & \in \mathbb{R}^m \\
        B & \in \mathbb{R}^{m \times n} \\
        b, y & \in \mathbb{R}^n \\
        A_i^{(l)}, C^{(l)}, X^{(l)}, Y^{(l)} & \in \mathbb{S}^{k^{(l)}} \\
    \end{aligned}
$$
    
## Interior-point method
In each iteration, the program solves the following deformed KKT conditions to determine the Newton step:
- Primal feasibility

$$ X^{(l)} = \sum_i x_i A_i^{(l)} - C^{(l)} $$

- Dual feasibility

$$ \sum_l tr(A_{\star}^{(l)} Y^{(l)}) + B y - c = 0 $$

- Complementarity

$$ X^{(l)} Y^{(l)} = \mu^{(l)} I $$

Mehrotra's predictor-corrector method is used to accelerate convergence. 

After a search direction is obtained, the step size is determined by requiring that $X$ and $Y$ remain positive.

## The feasibility problem
The function
```julia
findFeasible(A, C, B, b;
    β=0.1, Ωp=1, Ωd=1, γ=0.9, 
    ϵ_gap=1e-10, ϵ_primal=1e-10, ϵ_dual=1e-10,
    iterMax=200, prec=300, restart=true, minStep=1e-10)
```
determines whether the SDP above is feasible. Note that the arguments are basically the same as `sdp()` except no vector `c` for the objective function is needed. The function converts the feasibility problem to the following optimization problem:

$$
    \begin{aligned}
        \text{Minimize } \quad & t \\
        \text{subject to } \quad & X^{(l)} = \sum_i x_i A_i^{(l)} - C^{(l)} + t I\geq 0, \quad l = 1, 2, ..., L \\
        & B^T x = b 
    \end{aligned}
$$

If $t^* \geq 0$, the problem is infeasible; otherwise, the problem is feasible.

⚠️ Known issue: `findFeasible()` will not terminate if the feasible set is unbounded so that a minimal value of `t` does not exist, but it works fine when the problem is infeasible.

## Inputs

`prec`: arithemetic precision in base-10, which is equivalent to
```julia
setprecision(prec, base = 10)
```
The default value of the global variable `T` is `BigFloat`, which supports arbitrary precision arithmetic. 

If accuracy is not a concern, the user can manually set `T` to other arithmetic types for improved performance, say `Float64`:
```julia
setArithmeticType(Float64)
```

`c`: $m$-element `Vector{T}`

`A`: $L$-element `Vector{Array{T, 3}}`

`C`: $L$-element `Vector{Matrix{T}}`

`B`: $m$ x $n$ `Matrix{T}`

`b`: $n$-element `Vector{T}`

`β`: factor of reduction in μ in each step

`γ`: factor of reduction in the step size for backtracking line search

`Ωp` and `Ωd` are initial values for the matrices X and Y: $X = Ω_p I, Y = Ω_d I$

`restart`: `true` or `false`. If at any step in the iteration, the primal/dual step sizes are smaller than `minStep`, the program restarts with `Ωp` and `Ωd` rescaled by a factor of `OmegaStep`, until any of them exceeds `maxOmega`.

<!-- `mode`: can be either ```"opt"``` or ```"feas"```. -->

The iteration terminates if any of the following occurs:
- The function `sdp()` is used, and the duality gap, primal infeasibility, and dual infeasibility are below `ϵ_gap`, `ϵ_primal`, and `ϵ_dual`, respectively.
- The function `findFeasible()` is used, and the primal/dual infeasibilities reach their thresholds, with a certificate of $t^* > 0$ or $t^* < 0$ found.
- The number of iteration exceeds `iterMax`.


## Outputs
The function `sdp()` returns a dictionary with the following keys:
- "x": value of the variable `x`
- "X": value of the variable `X`
- "y": value of the variable `y`
- "Y": value of the variable `Y`
- "pObj": value of the primal objective function
- "dObj": value of the dual objective fucntion
- "status": reports the status of optimization. Can be either 
    * "Optimal"
    * "Feasible"
    * "Infeasible"
    * "Cannot reach optimality (feasibility) within `iterMax` iterations."

