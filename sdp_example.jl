include("SDPSolver.jl")

function test()
    T = BigFloat
    SDPSolver.setArithmeticType(T)
    A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
    A[1, 1, 1] = 1
    A[2, 2, 2] = 1
    C[1, 2], C[2, 1] = 1, 1
    A = [A]
    C = [C]
    c = [2, 3]
    B = Matrix{T}(undef, 2, 0)
    b = Array{T}(undef, 0)

    SDPSolver.setMode("opt")
    prob = SDPSolver.sdp(
        c, A, C, B, b; 
        β = 0.1, Ωp = 1, Ωd = 1, 
        ϵ_gap = 1e-20, ϵ_primal = 1e-20, ϵ_dual = 1e-20,
        iterMax = 100, prec = 50
    )
    println("\nStatus: ", prob["status"])
    println("Optimal value: ", prob["pObj"])
end

test()
