using SDPCSolver
using Test

@testset "SDPCSolver.jl" begin

    function test_float()
        @info "Testing Float64 arithmetic..."
        T = Float64
        setArithmeticType(T)
        A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
        A[1, 1, 1] = 1
        A[2, 2, 2] = 1
        C[1, 2], C[2, 1] = 1, 1
        A = [A]
        C = [C]
        c = [2, 3]
        B = Matrix{T}(undef, 2, 0)
        b = Array{T}(undef, 0)

        prob = sdp(c, A, C, B, b)
        println("\nStatus: ", prob["status"])
        println("Optimal value: ", prob["pObj"], "\n")
    end

    function test_bigfloat()
        @info "Testing BigFloat arithmetic..."
        T = BigFloat
        setArithmeticType(T)
        A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
        A[1, 1, 1] = 1
        A[2, 2, 2] = 1
        C[1, 2], C[2, 1] = 1, 1
        A = [A]
        C = [C]
        c = [2, 3]
        B = Matrix{T}(undef, 2, 0)
        b = Array{T}(undef, 0)

        prob = sdp(c, A, C, B, b; prec = 300)
        println("\nStatus: ", prob["status"])
        println("Optimal value: ", prob["pObj"], "\n")
    end

    test_float()
    test_bigfloat()

end
