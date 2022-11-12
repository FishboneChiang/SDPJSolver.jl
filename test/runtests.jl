using SDPJSolver
using Test

@testset "SDPJSolver.jl" begin

    function test_float()
        @info "Testing Float64 arithmetic..."
        T = Float64
        SDPJSolver.setArithmeticType(T)
        A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
        A[1, 1, 1] = 1
        A[2, 2, 2] = 1
        C[1, 2], C[2, 1] = 1, 1
        A = [A]
        C = [C]
        c = [2, 3]
        B = Matrix{T}(undef, 2, 0)
        b = Array{T}(undef, 0)

        prob = SDPJSolver.sdp(c, A, C, B, b)
        println("\nStatus: ", prob["status"])
        println("Optimal value: ", prob["pObj"], "\n")
    end

    function test_bigfloat()
        @info "Testing BigFloat arithmetic..."
        T = BigFloat
        SDPJSolver.setArithmeticType(T)
        A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
        A[1, 1, 1] = 1
        A[2, 2, 2] = 1
        C[1, 2], C[2, 1] = 1, 1
        A = [A]
        C = [C]
        c = [2, 3]
        B = Matrix{T}(undef, 2, 0)
        b = Array{T}(undef, 0)

        prob = SDPJSolver.sdp(c, A, C, B, b; prec=300)
        println("\nStatus: ", prob["status"])
        println("Optimal value: ", prob["pObj"], "\n")
    end

    function random_sdp()
        @info "Testing function findFeasible()"
        T = Float64
        SDPJSolver.setArithmeticType(T)
        m, n = 80, 0
        k = 20
        A1 = rand(T, m, k, k)
        A2 = rand(T, m, k, k)
        A3 = rand(T, m, k, k)
        C1 = rand(T, k, k)
        C2 = rand(T, k, k)
        C3 = rand(T, k, k)
        C1, C2, C3 = (x -> (transpose(x) + x) / 2).([C1, C2, C3])
        for i in 1:m
            A1[i, :, :] = (A1[i, :, :] + transpose(A1[i, :, :])) / 2
            A2[i, :, :] = (A2[i, :, :] + transpose(A2[i, :, :])) / 2
            A3[i, :, :] = (A3[i, :, :] + transpose(A3[i, :, :])) / 2
        end
        A, C = [A1, A2, A3], [C1, C2, C3]
        B = rand(T, m, n)
        b = zeros(T, n)
        c = rand(T, m)

        prob = SDPJSolver.findFeasible(A, C, B, b; Ωp = 1e4, Ωd = 1e4, β = 0.01, prec = 100, ϵ_dual = 1e-7, ϵ_primal = 1e-7, ϵ_gap = 1e-7)
        println("\np* = ", prob["pObj"], "\n\n", "Status: ", prob["status"])
    end

    test_float()
    test_bigfloat()
    random_sdp()

end
