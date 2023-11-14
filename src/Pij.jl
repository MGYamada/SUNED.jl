using MPI

using LinearAlgebra
using Combinatorics
using StatsFuns

const La = 2
const Lb = 2
const M = La * Lb
const N = 2M
const l = 3.0M + 1.0
const Λ = 100
const Nc = 4

function initialize()::Vector{Tuple{Int, Int}}
    NN = Tuple{Int, Int}[]
    for b in 1 : Lb, a in 1 : La
        X = (b - 1) * La + a
        B = 2X
        A = B - 1
        push!(NN, (B, A))
        push!(NN, (B, 2((mod1(b + 1, Lb) - 1) * La + a) - 1))
        push!(NN, (B, 2((b - 1) * La + mod1(a + 1, La)) - 1))
    end
    NN
end

function dcinit(D::Int, nproc::Int)
    @. D * (0 : nproc) ÷ nproc + 1
end

function τ!(y::Vector{ComplexF64}, x2::Vector{ComplexF64}, j::Int, axial::Matrix{Int8})
    @inbounds for i in 1 : length(y)
        ρ = 1.0 / axial[i, j]
        y[i] = sqrt(1.0 - ρ ^ 2) * x2[i] - ρ * y[i]
    end
end

function mτ!(y::Vector{ComplexF64}, x1::Vector{ComplexF64}, x2::Vector{ComplexF64}, j::Int, axial::Matrix{Int8})
    @inbounds for i in 1 : length(y)
        ρ = 1.0 / axial[i, j]
        y[i] -= sqrt(1.0 - ρ ^ 2) * x2[i] - ρ * x1[i]
    end
end

function gatheringtest!(j::Int, reverselist::Matrix{Int32}, counts::Matrix{Int32}, u1listaddress::Matrix{Int32}, sendbuf1::Vector{Int32}, recvbuf1::Vector{Int32})
    @inbounds for i in 1 : length(sendbuf1)
        sendbuf1[i] = u1listaddress[reverselist[i, j], j]
    end
    MPI.Alltoallv!(MPI.VBuffer(sendbuf1, counts[:, j]), MPI.VBuffer(recvbuf1, counts[:, j]), comm)
end

function gathering!(x::Vector{ComplexF64}, z1::Vector{ComplexF64}, j::Int, reverselist::Matrix{Int32}, counts::Matrix{Int32},
    recvbuf1list::Matrix{Int32}, sendbuf2::Vector{ComplexF64}, recvbuf2::Vector{ComplexF64})
    @inbounds for i in 1 : length(sendbuf2)
        sendbuf2[i] = x[reverselist[i, j]]
    end
    MPI.Alltoallv!(MPI.VBuffer(sendbuf2, counts[:, j]), MPI.VBuffer(recvbuf2, counts[:, j]), comm)
    @inbounds for i in 1 : length(recvbuf2)
        z1[recvbuf1list[i, j]] = recvbuf2[i]
    end
end

function H!(y::Vector{ComplexF64}, x::Vector{ComplexF64}, z1::Vector{ComplexF64}, z2::Vector{ComplexF64}, NNsorted::Vector{Tuple{Int, Int}},
    reverselist::Matrix{Int32}, counts::Matrix{Int32}, axial::Matrix{Int8}, recvbuf1list::Matrix{Int32}, sendbuf2::Vector{ComplexF64}, recvbuf2::Vector{ComplexF64})
    @. y = l * x
    for (a, b) in NNsorted
        if b - a == 1
            gathering!(x, z1, a, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
            mτ!(y, x, z1, a, axial)
        else
            z2 .= x
            gathering!(z2, z1, a, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
            τ!(z2, z1, a, axial)
            for j in a + 1 : b - 2
                gathering!(z2, z1, j, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
                τ!(z2, z1, j, axial)
            end
            for j in b - 1 : -1 : a + 1
                gathering!(z2, z1, j, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
                τ!(z2, z1, j, axial)
            end
            gathering!(z2, z1, a, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
            mτ!(y, z2, z1, a, axial)
        end
    end
end

function main(NN::Vector{Tuple{Int, Int}}, βs::Vector{Float64})::Tuple{Vector{Float64}, Vector{Float64}}
    NNsorted = map(nn -> (minimum(nn), maximum(nn)), NN)
    colors = filter(p -> length(p) <= Nc, integer_partitions(N))

    lnZ = zeros(Float64, length(βs), length(colors))
    lnZE = zeros(Float64, length(βs), length(colors))
    lnZEE = zeros(Float64, length(βs), length(colors))

    for (id, color) in enumerate(colors)
        if rank == 0
            println(color)
        end
        vertex = [[color]]
        edge = Vector{Vector{Int}}[]
        drop = Vector{Vector{Int}}[]
        for i in 1 : N
            push!(vertex, Vector{Vector{Int}}[])
            push!(edge, Vector{Int}[])
            push!(drop, Vector{Int}[])
            for j in 1 : length(vertex[i])
                push!(edge[i], Int[])
                push!(drop[i], Int[])
                for c in 1 : length(vertex[i][j])
                    color2 = copy(vertex[i][j])
                    position = c == 1 ? color2[c] : sum(color[1 : c - 1]) + color2[c]
                    if color2[c] > 1
                        if c == length(vertex[i][j]) || color2[c] > color2[c + 1]
                            color2[c] -= 1
                            test = findfirst(isequal(color2), vertex[i + 1])
                            if isnothing(test)
                                push!(vertex[i + 1], color2)
                                push!(edge[i][j], length(vertex[i + 1]))
                            else
                                push!(edge[i][j], test)
                            end
                            push!(drop[i][j], position)
                        end
                    else
                        if c == length(vertex[i][j])
                            popat!(color2, c)
                            test = findfirst(isequal(color2), vertex[i + 1])
                            if isnothing(test)
                                push!(vertex[i + 1], color2)
                                push!(edge[i][j], length(vertex[i + 1]))
                            else
                                push!(edge[i][j], test)
                            end
                            push!(drop[i][j], position)
                        end
                    end
                end
            end
        end
        b = [zeros(Int, length(vertex[i])) for i in 1 : N + 1]
        b[N + 1][1] = 1
        for i in N : -1 : 1
            for j in 1 : length(vertex[i])
                b[i][j] = sum(b[i + 1][edge[i][j]])
            end
        end
        f = deepcopy(edge)
        for i in 1 : N
            for j in 1 : length(edge[i])
                f[i][j][1] = 0
                bsum = 0
                for k in 2 : length(edge[i][j])
                    bsum += b[i + 1][edge[i][j][k - 1]]
                    f[i][j][k] = bsum
                end
            end
        end
        square = Vector{Vector{Vector{Tuple{Int, Int, Int8}}}}[]
        for i in 1 : N - 1
            push!(square, Vector{Vector{Tuple{Int, Int, Int8}}}[])
            for j in 1 : length(edge[i])
                push!(square[i], Vector{Tuple{Int, Int, Int8}}[])
                for k in 1 : length(edge[i][j])
                    push!(square[i][j], Tuple{Int, Int, Int8}[])
                    for m in 1 : length(edge[i + 1][edge[i][j][k]])
                        temp = drop[i + 1][edge[i][j][k]][m]
                        kprime = findfirst(isequal(temp), drop[i][j])
                        m1 = 0
                        m2 = 0
                        while temp > 0
                            m1 += 1
                            m2 = temp
                            temp -= color[m1]
                        end
                        temp = drop[i][j][k]
                        k1 = 0
                        k2 = 0
                        while temp > 0
                            k1 += 1
                            k2 = temp
                            temp -= color[k1]
                        end
                        if kprime != nothing
                            mprime = findfirst(isequal(drop[i][j][k]), drop[i + 1][edge[i][j][kprime]])
                            push!(square[i][j][k], (kprime, mprime, (k1 - m1) - (k2 - m2)))
                        else
                            push!(square[i][j][k], (k, m, (k1 - m1) - (k2 - m2)))
                        end
                    end
                end
            end
        end

        multiplicity = b[1][1]
        if rank == 0
            println(multiplicity)
        end
        grid = dcinit(multiplicity, size)
        chunk = grid[rank + 2] - grid[rank + 1]
        u1listgrid = zeros(Int16, chunk, N - 1)
        u1listaddress = zeros(Int32, chunk, N - 1)
        axial = zeros(Int8, chunk, N - 1)

        t1 = grid[rank + 1]
        r = t1 - 1
        j = 1
        codeword = Int[]
        jlist = Int[]
        for i in 1 : N
            k = searchsortedlast(f[i][j], r)
            push!(jlist, j)
            push!(codeword, k)
            r -= f[i][j][k]
            j = edge[i][j][k]
        end
        for h in 1 : chunk
            for i2 in 1 : N - 1
                kprime, mprime, ax = square[i2][jlist[i2]][codeword[i2]][codeword[i2 + 1]]
                codeword2 = copy(codeword)
                codeword2[i2] = kprime
                codeword2[i2 + 1] = mprime
                j = 1
                u = 0
                for i in 1 : N - 1 # fix later
                    k = codeword2[i]
                    u += f[i][j][k]
                    j = edge[i][j][k]
                end
                u1 = u + 1
                u1listgrid[h, i2] = (u1 * size - 1) ÷ multiplicity + 1 # fix later
                u1listaddress[h, i2] = u1 - grid[u1listgrid[h, i2]] + 1
                axial[h, i2] = ax
            end

            for i in N - 1 : -1 : 1
                codeword[i] += 1
                if codeword[i] <= length(edge[i][jlist[i]])
                    j = jlist[i]
                    for i2 in i : N - 1
                        jlist[i2] = j
                        j = edge[i2][j][codeword[i2]]
                    end
                    break
                else
                    codeword[i] = 1
                end
            end
            t1 += 1
        end

        reverselist = zeros(Int32, chunk, N - 1)
        counts = zeros(Int32, size, N - 1)
        for j in 1 : N - 1
            ilist = [Int32[] for k in 1 : size]
            for i in 1 : chunk
                push!(ilist[u1listgrid[i, j]], i)
            end
            reverselist[:, j] .= vcat(ilist...)
            @. counts[:, j] = length(ilist)
        end
        u1listgrid = Array{Int16}(undef, 0, 0)
        sendbuf1 = zeros(Int32, chunk)
        recvbuf1 = similar(sendbuf1)
        recvbuf1list = zeros(Int32, chunk, N - 1)
        for j in 1 : N - 1
            gatheringtest!(j, reverselist, counts, u1listaddress, sendbuf1, recvbuf1)
            recvbuf1list[:, j] .= recvbuf1
        end
        u1listaddress = Array{Int32}(undef, 0, 0)
        sendbuf1 = Int32[]
        recvbuf1 = Int32[]

        di = Vector{Int}[]
        for (i, c) in enumerate(color)
            push!(di, collect(Nc - i + 1 : Nc - i + c))
        end
        dim = Int(prod(big.(vcat(di...))) * multiplicity ÷ factorial(big(N)))
        if rank == 0
            println(dim)
        end

        ketk = randn(ComplexF64, chunk)
        temp0 = 1.0 / sqrt(MPI.Allreduce(dot(ketk, ketk), +, comm))
        ketk .*= temp0
        if rank == 0
            lognormk = 0.5(log(dim) + log(multiplicity))
        end
        ketk1 = similar(ketk)
        z1 = similar(ketk)
        z2 = similar(ketk)
        sendbuf2 = similar(ketk)
        recvbuf2 = similar(sendbuf2)
        H!(ketk1, ketk, z1, z2, NNsorted, reverselist, counts, axial, recvbuf1list, sendbuf2, recvbuf2)
        temp1 = sqrt(MPI.Allreduce(dot(ketk1, ketk1), +, comm))
        if rank == 0
            lognormk1 = lognormk + log(temp1)
        end
        temp1 = 1.0 / temp1
        ketk1 .*= temp1

        if rank == 0
            sumlogm = 0.0
        end
        kk1 = MPI.Reduce(real(dot(ketk, ketk1)), +, 0, comm)
        @time for k in 0 : Λ - 1
            ketk .= ketk1
            H!(ketk1, ketk, z1, z2, NNsorted, reverselist, counts, axial, recvbuf1list, sendbuf2, recvbuf2)
            temp2 = sqrt(MPI.Allreduce(dot(ketk1, ketk1), +, comm))
            if rank == 0
                lognormk2 = lognormk1 + log(temp2)
            end
            temp2 = 1.0 / temp2
            ketk1 .*= temp2
            k1k2 = MPI.Reduce(real(dot(ketk, ketk1)), +, 0, comm)
            if rank == 0
                for (i, β) in enumerate(βs)
                    if k == 0
                        temp3 = 0.0
                        lnZ[i, id] = 2lognormk
                        lnZE[i, id] = lognormk + lognormk1 + log(kk1)
                        lnZEE[i, id] = 2lognormk1
                    else
                        temp3 = 2k * log(β) - sumlogm
                        lnZ[i, id] += log1pexp(-lnZ[i, id] + (temp3 + 2lognormk))
                        lnZE[i, id] += log1pexp(-lnZE[i, id] + (temp3 + lognormk + lognormk1) + log(kk1))
                        lnZEE[i, id] += log1pexp(-lnZEE[i, id] + (temp3 + 2lognormk1))
                    end
                    temp3 += log(β) - log(2k + 1)
                    lnZ[i, id] += log1pexp(-lnZ[i, id] + (temp3 + lognormk + lognormk1) + log(kk1))
                    lnZE[i, id] += log1pexp(-lnZE[i, id] + (temp3 + 2lognormk1))
                    lnZEE[i, id] += log1pexp(-lnZEE[i, id] + (temp3 + lognormk1 + lognormk2) + log(k1k2))
                end
                kk1 = k1k2
                lognormk = lognormk1
                lognormk1 = lognormk2
                sumlogm += log(2k + 1) + log(2k + 2)
            end
        end
    end

    lnZsum = lnZ[:, 1]
    lnZEsum = lnZE[:, 1]
    lnZEEsum = lnZEE[:, 1]
    if rank == 0
        for id in 2 : length(colors)
            @. lnZsum += log1pexp(-lnZsum + lnZ[:, id])
            @. lnZEsum += log1pexp(-lnZEsum + lnZE[:, id])
            @. lnZEEsum += log1pexp(-lnZEEsum + lnZEE[:, id])
        end
    end

    @. βs ^ 2 * (exp(lnZEEsum - lnZsum) - exp(lnZEsum - lnZsum) ^ 2), @. exp(lnZEsum - lnZsum)
end

MPI.Init()

comm = MPI.COMM_WORLD
size = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

βs = 10 .^ (-4.0 : 0.01 : 4.0)
Cv1, Cv2 = main(initialize(), βs)
if rank == 0
    println(Cv1)
    println(Cv2)
end

MPI.Finalize()