using LinearAlgebra, Base.Threads, Printf

T::Type = BigFloat
mode::String = "opt"

function setArithmeticType(type)
    global T = type
end

function setMode(str)
    if str ∈ ["opt", "feas"]
        global mode = str
    else
        @error "Mode should be either \"opt\" or \"feas\"!"
    end
end

function getResidue(μ, x, X, y, Y, c, A, C, B, b)
    m, L = length(x), length(A)
    P = [sum([A[l][i, :, :] * x[i] for i in 1:m]) - X[l] - C[l] for l in 1:L]
    p = b - transpose(B) * x
    d = c - Array([sum([sum(A[l][i, :, :] .* Y[l]) for l in 1:L]) for i in 1:m]) - B * y
    R = [μ[l] * I - X[l] * Y[l] for l in 1:L]
    return P, p, d, R
end

function NewtonStep(β, μ, x, X, y, Y, c, A, C, B, b)
    m, L, n = length(x), length(A), length(y)

    # calculate residue
    P, p, d, R = getResidue(μ, x, X, y, Y, c, A, C, B, b)

    # calculate Schur complement
    begin
        S = [zeros(T, m, m) for l in 1:L]
        invX = Array{Matrix{T}}(undef, L)
        @threads for l in 1:L
            invX[l] = X[l] \ I
        end
        @threads for i in 1:m
            for l in 1:L
                SS = Y[l] * A[l][i, :, :] * invX[l]
                for j in i:m
                    S[l][i, j] = sum(SS .* A[l][j, :, :])
                    S[l][j, i] = S[l][i, j]
                end
            end
        end
    end

    # Predictor
    begin
        M = vcat(hcat(sum(S), -B), hcat(transpose(B), zeros(n, n)))
        v = zeros(T, m)
        Z = Array{Matrix{T}}(undef, m)
        @threads for l in 1:L
            Z[l] = X[l] \ (P[l] * Y[l] - R[l])
        end
        @threads for i in 1:m
            for l in 1:L
                v[i] += sum(A[l][i, :, :] .* Z[l])
            end
        end
        dxdy = M \ vcat(-d - v, p)
        dx, dy = dxdy[1:m], dxdy[m+1:m+n]
        dX = P + [sum([A[l][i, :, :] * dx[i] for i in 1:m]) for l in 1:L]
        dY = X .\ (R - dX .* Y)
        dY = (dY + transpose.(dY)) / 2
    end

    # Corrector
    begin
        r = [sum((X[l] + dX[l]) .* (Y[l] + dY[l])) / μ[l] / size(X[l])[1] for l in 1:L]
        γ = [max(r[l] < 1 ? r[l]^2 : r[l], β) for l in 1:L]
        if all(isposdef.(X .+ dX)) && all(isposdef.(Y .+ dY))
            γ = [min(γ[l], 1) for l in 1:L]
        end
        R = [γ[l] * μ[l] * I - X[l] * Y[l] - dX[l] * dY[l] for l in 1:L]

        v = zeros(T, m)
        @threads for l in 1:L
            Z[l] = X[l] \ (P[l] * Y[l] - R[l])
        end
        @threads for i in 1:m
            for l in 1:L
                v[i] += sum(A[l][i, :, :] .* Z[l])
            end
        end
        dxdy = M \ vcat(-d - v, p)
        dx, dy = dxdy[1:m], dxdy[m+1:m+n]
        dX = P + [sum([A[l][i, :, :] * dx[i] for i in 1:m]) for l in 1:L]
        dY = X .\ (R - dX .* Y)
        dY = (dY + transpose.(dY)) / 2
    end

    p_res, d_res = max([max(abs.(P[l])...) for l in 1:L]..., abs.(p)...), max(abs.(d)...)

    return p_res, d_res, dx, dX, dy, dY
end

function sdp(c, A, C, B, b;
    β=0.1, Ωp=1, Ωd=1,
    ϵ_gap=1e-10, ϵ_primal=1e-10, ϵ_dual=1e-10,
    iterMax=200, prec=300)

    # Set arithmetic type and precision
    if T == BigFloat
        setprecision(prec, base=10)
    end
    # c, A, C, B, b = T.(c), T.(A), T.(C), T.(B), T.(b)

    # Initialize variables
    L, m, n = length(A), size(A[1])[1], length(b)
    x, y = zeros(T, m), zeros(T, n)
    X, Y, μ = Array{Matrix{T}}(undef, L), Array{Matrix{T}}(undef, L), Array{T}(undef, L)
    for l in 1:L
        X[l], Y[l] = Matrix(Ωp * I, size(A[l])[2:3]), Matrix(Ωd * I, size(A[l])[2:3])
        μ[l] = sum(X[l] .* Y[l]) / size(X[l])[1]
    end

    iter = 0
    primal_obj = transpose(c) * x
    dual_obj = sum(tr.(C .* Y)) + transpose(b) * y
    dual_gap = primal_obj - dual_obj
    P, p, d, R = getResidue(μ, x, X, y, Y, c, A, C, B, b)
    p_res, d_res = max([max(abs.(P[l])...) for l in 1:L]..., abs.(p)...), max(abs.(d)...)

    println("iter\tp-Obj\t\td-Obj\t\tgap\t\tp-Res\t\td-Res\t\tstep\t\ttime")
    println("=====================================================================================================================")
    @printf "%d\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n" iter primal_obj dual_obj dual_gap p_res d_res
    if p_res < ϵ_primal && d_res < ϵ_dual
        if mode == "opt" && 0 < dual_gap < ϵ_gap
            return Dict("x" => x, "X" => X, "y" => y, "Y" => Y, "pObj" => primal_obj, "dObj" => dual_obj, "status" => "Optimal")
        end
        if mode == "feas"
            if dual_obj <= primal_obj <= 0
                return Dict("x" => x, "X" => X, "y" => y, "Y" => Y, "pObj" => primal_obj, "dObj" => dual_obj, "status" => "Feasible")
            end
            if primal_obj >= dual_obj >= 0
                return Dict("x" => x, "X" => X, "y" => y, "Y" => Y, "pObj" => primal_obj, "dObj" => dual_obj, "status" => "Infeasible")
            end
        end
    end

    while iter < iterMax
        t1 = time()
        p_res, d_res, dx, dX, dy, dY = NewtonStep(β, μ, x, X, y, Y, c, A, C, B, b)

        # Line search
        t = 1
        while true
            X_new, Y_new = X + t * dX, Y + t * dY
            if !(all(isposdef.(X_new)) && all(isposdef.(Y_new)))
                t *= 0.9
                continue
            end
            x, y = x + t * dx, y + t * dy
            X, Y = X_new, Y_new
            break
        end
        t2 = time()

        primal_obj = transpose(c) * x
        dual_obj = sum(tr.(C .* Y)) + transpose(b) * y
        dual_gap = primal_obj - dual_obj
        iter += 1
        @printf "%d\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n" iter primal_obj dual_obj dual_gap p_res d_res t t2 - t1

        if p_res < ϵ_primal && d_res < ϵ_dual
            if mode == "opt" && 0 < dual_gap < ϵ_gap
                return Dict("x" => x, "X" => X, "y" => y, "Y" => Y, "pObj" => primal_obj, "dObj" => dual_obj, "status" => "Optimal")
            end
            if mode == "feas"
                if dual_obj <= primal_obj < 0
                    return Dict("x" => x, "X" => X, "y" => y, "Y" => Y, "pObj" => primal_obj, "dObj" => dual_obj, "status" => "Feasible")
                end
                if primal_obj >= dual_obj > 0
                    return Dict("x" => x, "X" => X, "y" => y, "Y" => Y, "pObj" => primal_obj, "dObj" => dual_obj, "status" => "Infeasible")
                end
            end
        end

        μ = β * tr.(X .* Y) ./ [size(XX)[1] for XX in X]
    end

    return Dict("x" => x, "X" => X, "y" => y, "Y" => Y, "pObj" => primal_obj, "dObj" => dual_obj, "status" => "Cannot reach optimality (feasibility) within $(iterMax) iterations.")

end

function findFeasible(A, C, B, b;
    β=0.1, Ωp=1, Ωd=1,
    ϵ_gap=1e-10, ϵ_primal=1e-10, ϵ_dual=1e-10,
    iterMax=200, prec=300)

    # Initialize variables
    L, m, n = length(A), size(A[1])[1], length(b)

    # sdp parameters
    AA = Array{Any}(undef, L)
    for l in 1:L
        k = size(A[l])[2]
        AA[l] = vcat(reshape(Matrix{T}(I, k, k), 1, k, k), A[l])
    end
    cc = [i == 1 ? 1 : 0 for i in 1:m+1]
    BB = vcat(zeros(T, 1, n), B)

    setMode("feas")

    prob = sdp(cc, AA, C, BB, b;
        β=β, Ωp=Ωp, Ωd=Ωd,
        ϵ_gap=ϵ_gap, ϵ_primal=ϵ_primal, ϵ_dual=ϵ_dual, iterMax=iterMax, prec=prec)

    setMode("opt")

    return prob

end