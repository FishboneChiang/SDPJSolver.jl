using LinearAlgebra, Base.Threads, Printf, SparseArrays

T::Type = BigFloat
mode::String = "opt"
sparseMode::Bool = false

function setSparseMode(arg)
    if arg == true
        println("Sparse mode ON")
        global sparseMode = true
    elseif arg == false
        println("Sparse mode OFF")
        global sparseMode = false
    else
        println("sparseMode should be true or false!")
    end
end

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

#=====================================================================
    These functions are for the PRIMAL-DUAL interior-point method.
=====================================================================#

#=============
    Dense 
=============#

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
            @inbounds invX[l] = X[l] \ I
        end
        @threads for i in 1:m
            for l in 1:L
                @inbounds SS = Y[l] * A[l][i, :, :] * invX[l]
                for j in i:m
                    @inbounds S[l][i, j] = sum(SS .* A[l][j, :, :])
                    @inbounds S[l][j, i] = S[l][i, j]
                end
            end
        end
    end

    # Predictor
    begin
        M = vcat(hcat(sum(S), -B), hcat(transpose(B), zeros(n, n)))
        v = zeros(T, m)
        Z = Array{Matrix{T}}(undef, L)
        @threads for l in 1:L
            @inbounds Z[l] = invX[l] * (P[l] * Y[l] - R[l])
        end
        @threads for i in 1:m
            for l in 1:L
                @inbounds v[i] += sum(A[l][i, :, :] .* Z[l])
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
            @inbounds Z[l] = X[l] \ (P[l] * Y[l] - R[l])
        end
        @threads for i in 1:m
            for l in 1:L
                @inbounds v[i] += sum(A[l][i, :, :] .* Z[l])
            end
        end
        dxdy = M \ vcat(-d - v, p)
        dx, dy = dxdy[1:m], dxdy[m+1:m+n]
        dX = P + [sum([A[l][i, :, :] * dx[i] for i in 1:m]) for l in 1:L]
        dY = X .\ (R - dX .* Y)
        dY = (dY + transpose.(dY)) / 2
    end

    p_res, d_res = max([max(abs.(P[l])...) for l in 1:L]..., abs.(p)...), max(abs.(d)...)

    GC.gc()

    return p_res, d_res, dx, dX, dy, dY
end

#=============
    Sparse 
=============#

function NewtonStepSparse(β, μ, x, X, y, Y, c, A, AA, C, B, b)
    m, L, n = length(x), length(AA), length(y)

    # calculate residue
    P, p, d, R = getResidue(μ, x, X, y, Y, c, A, C, B, b)

    # calculate Schur complement
    begin
        S = [zeros(T, m, m) for l in 1:L]
        invX = Array{Matrix{T}}(undef, L)
        @threads for l in 1:L
            @inbounds invX[l] = X[l] \ I
        end
        for l in 1:L
            SS1, SS2 = Array{Matrix{T}}(undef, m), Array{Matrix{T}}(undef, m)
            @threads for i in 1:m
                SS1[i] = Y[l] * AA[l][i]
                SS2[i] = AA[l][i] * invX[l]
            end
            @threads for i in 1:m
                for j in i:m
                    S[l][i, j] = sum(SS1[i] .* SS2[j])
                    S[l][j, i] = S[l][i, j]
                end
            end
        end
    end

    # Predictor
    begin
        M = vcat(hcat(sum(S), -B), hcat(transpose(B), zeros(n, n)))
        v = zeros(T, m)
        Z = Array{Matrix{T}}(undef, L)
        @threads for l in 1:L
            @inbounds Z[l] = invX[l] * (P[l] * Y[l] - R[l])
        end
        @threads for i in 1:m
            for l in 1:L
                @inbounds v[i] += sum(AA[l][i] .* Z[l])
            end
        end
        dxdy = M \ vcat(-d - v, p)
        dx, dy = dxdy[1:m], dxdy[m+1:m+n]
        dX = P + [sum([A[l][i,:,:] * dx[i] for i in 1:m]) for l in 1:L]
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
            @inbounds Z[l] = X[l] \ (P[l] * Y[l] - R[l])
        end
        @threads for i in 1:m
            for l in 1:L
                @inbounds v[i] += sum(AA[l][i] .* Z[l])
            end
        end
        dxdy = M \ vcat(-d - v, p)
        dx, dy = dxdy[1:m], dxdy[m+1:m+n]
        dX = P + [sum([A[l][i,:,:] * dx[i] for i in 1:m]) for l in 1:L]
        dY = X .\ (R - dX .* Y)
        dY = (dY + transpose.(dY)) / 2
    end

    p_res, d_res = max([max(abs.(P[l])...) for l in 1:L]..., abs.(p)...), max(abs.(d)...)

    GC.gc()

    return p_res, d_res, dx, dX, dy, dY
end

function sdp(c, A, C, B, b;
    β=0.1, Ωp=1, Ωd=1,
    ϵ_gap=1e-10, ϵ_primal=1e-10, ϵ_dual=1e-10,
    iterMax=200, prec=300, 
    restart = true, minStep = 1e-15, maxOmega = 1e50)

    # Set arithmetic type and precision
    if T == BigFloat
        setprecision(prec, base=10)
    end
    # c, A, C, B, b = T.(c), T.(A), T.(C), T.(B), T.(b)

    @label start
    if Ωp > maxOmega || Ωd > maxOmega
        return Dict("x" => x, "X" => X, "y" => y, "Y" => Y, "pObj" => primal_obj, "dObj" => dual_obj, "status" => "maxOmega exceeded!")
    end

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

    println("iter\tp-Obj\t\td-Obj\t\tgap\t\tp-Res\t\td-Res\t\tp-step\t\td-step\t\ttime")
    println("===================================================================================================================================")
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

    AA = []
    if sparseMode
        for l in 1:L
            push!(AA, [sparse(A[l][i,:,:]) for i in 1:m])
        end
    end

    while iter < iterMax
        t1 = time()
        if sparseMode
            p_res, d_res, dx, dX, dy, dY = NewtonStepSparse(β, μ, x, X, y, Y, c, A, AA, C, B, b)
        else
            p_res, d_res, dx, dX, dy, dY = NewtonStep(β, μ, x, X, y, Y, c, A, C, B, b)
        end

        # Line search
        tX, tY = 1, 1
        while true
            # restart if the step sizes are too small
            if restart && tX < minStep && tY < minStep
                println("Step size too small! Restart!")
                Ωp *= 1e5
                Ωd *= 1e5
                @goto start
            end
            # 
            X_new, Y_new = X + tX * dX, Y + tY * dY
            if !(all(isposdef.(X_new)))
                tX *= 0.9
                continue
            end
            if !(all(isposdef.(Y_new)))
                tY *= 0.9
                continue
            end
            x, y = x + tX * dx, y + tY * dy
            X, Y = X_new, Y_new
            break
        end
        t2 = time()

        primal_obj = transpose(c) * x
        dual_obj = sum(tr.(C .* Y)) + transpose(b) * y
        dual_gap = primal_obj - dual_obj
        iter += 1
        @printf "%d\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n" iter primal_obj dual_obj dual_gap p_res d_res tX tY t2 - t1

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

function sdp(c, A, C, B, b, x0, X0, y0, Y0;
    β=0.1,
    ϵ_gap=1e-10, ϵ_primal=1e-10, ϵ_dual=1e-10,
    iterMax=200, prec=300)

    # Set arithmetic type and precision
    if T == BigFloat
        setprecision(prec, base=10)
    end
    # c, A, C, B, b = T.(c), T.(A), T.(C), T.(B), T.(b)

    # Initialize variables
    L, m, n = length(A), size(A[1])[1], length(b)
    x, y = x0, y0
    X, Y, μ = X0, Y0, Array{T}(undef, L)
    for l in 1:L
        μ[l] = sum(X[l] .* Y[l]) / size(X[l])[1]
    end

    # Check positivity
    if !(all(isposdef.(X)) && all(isposdef.(Y)))
        @error "Initial point X or Y is not positive!"
        return
    end

    iter = 0
    primal_obj = transpose(c) * x
    dual_obj = sum(tr.(C .* Y)) + transpose(b) * y
    dual_gap = primal_obj - dual_obj
    P, p, d, R = getResidue(μ, x, X, y, Y, c, A, C, B, b)
    p_res, d_res = max([max(abs.(P[l])...) for l in 1:L]..., abs.(p)...), max(abs.(d)...)

    println("iter\tp-Obj\t\td-Obj\t\tgap\t\tp-Res\t\td-Res\t\tp-step\t\td-step\t\ttime")
    println("===================================================================================================================================")
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
        tX, tY = 1, 1
        while true
            X_new, Y_new = X + tX * dX, Y + tY * dY
            if !(all(isposdef.(X_new)))
                tX *= 0.9
                continue
            end
            if !(all(isposdef.(Y_new)))
                tY *= 0.9
                continue
            end
            x, y = x + tX * dx, y + tY * dy
            X, Y = X_new, Y_new
            break
        end
        t2 = time()

        primal_obj = transpose(c) * x
        dual_obj = sum(tr.(C .* Y)) + transpose(b) * y
        dual_gap = primal_obj - dual_obj
        iter += 1
        @printf "%d\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n" iter primal_obj dual_obj dual_gap p_res d_res tX tY t2 - t1

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

function findFeasible(A, C, B, b, x0, X0, y0, Y0;
    β=0.1,
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

    prob = sdp(cc, AA, C, BB, b, x0, X0, y0, Y0;
        β=β,
        ϵ_gap=ϵ_gap, ϵ_primal=ϵ_primal, ϵ_dual=ϵ_dual, iterMax=iterMax, prec=prec)

    setMode("opt")

    return prob

end

#================================(EXPERIMENTAL)================================
    These functions are for the BARRIER interior-point method (with BFGS).
==============================================================================#

function f1(A, C, x)
    L, m = length(A), size(A[1])[1]
    X = Array{Matrix{T}}(undef, L)
    @threads for l in 1:L
        X[l] = -C[l] + sum([x[i] * A[l][i, :, :] for i in 1:m])
    end
    return X
end

function df(c, μ, A, C, x)
    L, m = length(A), size(A[1])[1]
    invX = f1(A, C, x)
    @threads for l in 1:L
        invX[l] = invX[l] \ I
    end
    g = T.(c)
    @threads for i in 1:m
        for l in 1:L
            g[i] -= μ * sum(A[l][i, :, :] .* invX[l])
        end
    end
    return g
end

function f(c, μ, A, C, x)
    f = c' * x
    X = f1(A, C, x)
    f -= μ * sum(log.(det.(X)))
end

function sdpBFGS(c, A, C, x0;
    μ=1, β=0.1,
    ϵ_gap=1e-7, ϵ=1e-10, iterMax=100, prec=300)

    # Set arithmetic type and precision
    if T == BigFloat
        setprecision(prec, base=10)
    end
    # c, A, C, x0 = T.(c), T.(A), T.(C), T.(x0)

    # Initialize variables
    L, m = length(A), size(A[1])[1]
    x = x0
    X = f1(A, C, x)
    H = Matrix{T}(I, m, m)

    # Check feasibility
    if !(all(isposdef.(X)))
        @error "Initial point not feasible!"
        return
    end

    primal_obj = c' * x
    X = f1(A, C, x)
    dual_obj = μ * sum([sum(C[l] .* (X[l] \ I)) for l in 1:L])
    dual_gap = primal_obj - dual_obj
    println("μ\t\tp-Obj\t\td-Obj\t\tgap\t\tres\t\tsteps\ttime")
    println("===================================================================================================")

    while true

        iter = 0

        t1 = time()

        while true
            dx = -H * df(c, μ, A, C, x)
            # Line search
            t = 1
            while !(all(isposdef.(f1(A, C, x + t * dx))))
                t *= 0.9
                if t < 1e-5
                    @error "Step size too small!"
                    return
                end
            end
            dx = t * dx
            x_new = x + dx
            y = df(c, μ, A, C, x_new) - df(c, μ, A, C, x)
            x = x_new
            M = I - (dx * y') / (y' * dx)
            H = M * H * M' + (dx * dx') / (y' * dx)

            if norm(df(c, μ, A, C, x)) < ϵ
                t2 = time()
                primal_obj = c' * x
                X = f1(A, C, x)
                dual_obj = μ * sum([sum(C[l] .* (X[l] \ I)) for l in 1:L])
                dual_gap = primal_obj - dual_obj
                res = max(abs.(df(c, μ, A, C, x))...)
                @printf "%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%d\t%.5E\n" μ primal_obj dual_obj dual_gap res iter t2 - t1
                break
            end

            if iter >= iterMax
                println("Cannot reach optimality within $(iterMax) iterations!")
                return
            end

            iter += 1
        end

        if 0 < dual_gap < ϵ_gap
            break
        end

        μ *= β

    end

    return Dict("x" => x, "X" => X, "pObj" => primal_obj, "dObj" => dual_obj, "status" => "Optimal")

end

function findFeasibleBFGS(c, A, C;
    μ=1, β=0.1,
    ϵ_gap=1e-10, ϵ=1e-5, iterMax=100, prec=300)

    # Initialize variables
    L, m = length(A), size(A[1])[1]

    # sdp parameters
    AA = Array{Any}(undef, L)
    for l in 1:L
        k = size(A[l])[2]
        AA[l] = vcat(reshape(Matrix{T}(I, k, k), 1, k, k), A[l])
    end
    cc = [i == 1 ? 1 : 0 for i in 1:m+1]
    t0 = 1
    while !(all(isposdef.([t0 * I - C[l] for l in 1:L]))) t0 *= 2 end
    println("Initial point found: t0 = $(t0)")
    x0 = Array([t0, zeros(T, m)...])

    prob = sdpBFGS(cc, AA, C, x0;
       μ = μ, β=β, ϵ_gap=ϵ_gap, ϵ = ϵ, iterMax=iterMax, prec=prec)

    return prob

end