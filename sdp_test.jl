include("SDPSolver.jl")

T = BigFloat

function test()

    A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
    A[1, 1, 1] = 1
    A[2, 2, 2] = 1
    C[1, 2], C[2, 1] = 1, 1
    A = [A]
    C = [C]
    c = [2, 3]
    B = Matrix{T}(undef, 2, 0)
    b = Array{T}(undef, 0)
    # B = [1 0 0; 0 1 0] |> transpose
    # b = [1, 1]

    prob = sdp(100, c, A, C, B, b, 0.1, 1e-50, 1e-50, 1e-50, 200)
    println("\np* = ", prob["pObj"], "\n\n", "Status: ", prob["status"])
end

test()