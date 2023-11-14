using MPI

using LinearAlgebra
using Combinatorics

const N = 6
const Nc = 4

function initialize()::Vector{Tuple{Tuple{Int, Int}, Float64}}
    # vcat(map(z -> z[1] % 6 == 0 ? (z, 0.1) : (z, 1.0), collect(zip(1 : 23, 2 : 24))), collect(zip(zip(2 : 2 : 4, 11 : -2 : 9), [0.1, 0.1])), collect(zip(zip(8 : 2 : 10, 17 : -2 : 15), [0.1, 0.1])), collect(zip(zip(14 : 2 : 16, 23 : -2 : 21), [0.1, 0.1])), collect(zip(zip(1 : 6 : 19, 6 : 6 : 24), ones(4))))
    # vcat(map(z -> z[1] % 4 == 0 ? (z, 0.1) : (z, 1.0), collect(zip(1 : 23, 2 : 24))), collect(zip(zip(2 : 4 : 18, 7 : 4 : 23), 0.1ones(5))), collect(zip(zip(1 : 4 : 21, 4 : 4 : 24), ones(6))))
    [((1, 2), 1.0), ((2, 3), 1.0), ((3, 4), 1.0), ((4, 5), 1.0), ((5, 6), 1.0), ((6, 1), 1.0)]
end

function dcinit(D::Int, nproc::Int)
    @. D * (0 : nproc) ÷ nproc + 1
end

function τ!(y::Vector{Float64}, x2::Vector{Float64}, k1::Int, axial::Matrix{Int8})
    @inbounds for i in 1 : length(y)
        ρ = 1.0 / axial[i, k1]
        y[i] = sqrt(1.0 - ρ ^ 2) * x2[i] - ρ * y[i]
    end
end

function pτ!(y::Vector{Float64}, x1::Vector{Float64}, x2::Vector{Float64}, k1::Int, axial::Matrix{Int8}, w::Float64)
    @inbounds for i in 1 : length(y)
        ρ = 1.0 / axial[i, k1]
        y[i] += w * (sqrt(1.0 - ρ ^ 2) * x2[i] - ρ * x1[i])
    end
end

function gatheringtest!(k1::Int, reverselist::Matrix{Int32}, counts::Matrix{Cint}, u1listaddress::Matrix{Int32}, sendbuf1::Vector{Int32}, recvbuf1::Vector{Int32})
    @inbounds for i in 1 : length(sendbuf1)
        sendbuf1[i] = u1listaddress[reverselist[i, k1], k1]
    end
    MPI.Alltoallv!(MPI.VBuffer(sendbuf1, counts[:, k1]), MPI.VBuffer(recvbuf1, counts[:, k1]), comm)
end

function gathering!(x::Vector{Float64}, z1::Vector{Float64}, k1::Int, reverselist::Matrix{Int32}, counts::Matrix{Cint},
    recvbuf1list::Matrix{Int32}, sendbuf2::Vector{Float64}, recvbuf2::Vector{Float64})
    @inbounds for i in 1 : length(sendbuf2)
        sendbuf2[i] = x[reverselist[i, k1]]
    end
    MPI.Alltoallv!(MPI.VBuffer(sendbuf2, counts[:, k1]), MPI.VBuffer(recvbuf2, counts[:, k1]), comm)
    @inbounds for i in 1 : length(recvbuf2)
        z1[recvbuf1list[i, k1]] = recvbuf2[i]
    end
end

function Pij!(a::Int, b::Int, y::Vector{Float64}, x::Vector{Float64}, z1::Vector{Float64}, z2::Vector{Float64},
    reverselist::Matrix{Int32}, counts::Matrix{Cint}, axial::Matrix{Int8}, recvbuf1list::Matrix{Int32}, sendbuf2::Vector{Float64}, recvbuf2::Vector{Float64})
    y .= 0.0
    if b - a == 1
        gathering!(x, z1, a, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
        pτ!(y, x, z1, a, axial, 1.0)
    else
        z2 .= x
        gathering!(z2, z1, a, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
        τ!(z2, z1, a, axial)
        for k1 in a + 1 : b - 2
            gathering!(z2, z1, k1, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
            τ!(z2, z1, k1, axial)
        end
        for k1 in b - 1 : -1 : a + 1
            gathering!(z2, z1, k1, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
            τ!(z2, z1, k1, axial)
        end
        gathering!(z2, z1, a, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
        pτ!(y, z2, z1, a, axial, 1.0)
    end
end

function H!(y::Vector{Float64}, x::Vector{Float64}, z1::Vector{Float64}, z2::Vector{Float64}, NNsorted::Vector{Tuple{Tuple{Int, Int}, Float64}},
    reverselist::Matrix{Int32}, counts::Matrix{Cint}, axial::Matrix{Int8}, recvbuf1list::Matrix{Int32}, sendbuf2::Vector{Float64}, recvbuf2::Vector{Float64})
    y .= 0.0
    for ((a, b), w) in NNsorted
        if b - a == 1
            gathering!(x, z1, a, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
            pτ!(y, x, z1, a, axial, w)
        else
            z2 .= x
            gathering!(z2, z1, a, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
            τ!(z2, z1, a, axial)
            for k1 in a + 1 : b - 2
                gathering!(z2, z1, k1, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
                τ!(z2, z1, k1, axial)
            end
            for k1 in b - 1 : -1 : a + 1
                gathering!(z2, z1, k1, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
                τ!(z2, z1, k1, axial)
            end
            gathering!(z2, z1, a, reverselist, counts, recvbuf1list, sendbuf2, recvbuf2)
            pτ!(y, z2, z1, a, axial, w)
        end
    end
end

function main(id::Int, NN::Vector{Tuple{Tuple{Int, Int}, Float64}})
    NNsorted = map(nn -> ((minimum(nn[1]), maximum(nn[1])), nn[2]), NN)
    colors = filter(p -> length(p) <= Nc, integer_partitions(N))

    color = colors[id]
    if rank == 0
        println(color)
    end
    vertex = [[color]]
    edge = Vector{Vector{Int}}[]
    drop = Vector{Vector{Int}}[]
    for i in 1 : N - 1
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
                        if test == nothing
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
                        if test == nothing
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
    b = [zeros(Int, length(vertex[i])) for i in 1 : N]
    b[N][1] = 1
    for i in N - 1 : -1 : 1
        for j in 1 : length(vertex[i])
            b[i][j] = sum(b[i + 1][edge[i][j]])
        end
    end
    f = deepcopy(edge)
    for i in 1 : N - 1
        for j in 1 : length(edge[i])
            f[i][j][1] = 0
            bsum = 0
            for k in 2 : length(edge[i][j])
                bsum += b[i + 1][edge[i][j][k - 1]]
                f[i][j][k] = bsum
            end
        end
    end

    dim = b[1][1]
    if rank == 0
        println(dim)
    end
    grid = dcinit(dim, Ncpu)
    chunk = grid[rank + 2] - grid[rank + 1]
    u1listgrid = zeros(Int16, chunk, N - 1)
    u1listaddress = zeros(Int32, chunk, N - 1)
    axial = zeros(Int8, chunk, N - 1)
    for h in 1 : chunk
        t1 = grid[rank + 1] - 1 + h
        r = t1 - 1
        j = 1
        St = ones(Int, N)
        # codeword = Int[]
        for i in 1 : N - 1
            k = searchsortedlast(f[i][j], r)
            # push!(codeword, k)
            St[drop[i][j][k]] = N + 1 - i
            r -= f[i][j][k]
            j = edge[i][j][k]
        end
        mlist = zeros(Int, N)
        m1list = zeros(Int, N)
        m2list = zeros(Int, N)
        for k2 in 1 : N
            mlist[k2] = findfirst(isequal(k2), St)
            temp = mlist[k2]
            while temp > 0
                m1list[k2] += 1
                m2list[k2] = temp
                temp -= color[m1list[k2]]
            end
        end
        for k1 in 1 : N - 1
            m = mlist[k1]
            m1 = m1list[k1]
            m2 = m2list[k1]
            n = mlist[k1 + 1]
            n1 = m1list[k1 + 1]
            n2 = m2list[k1 + 1]
            if m1 == n1
                u1listgrid[h, k1] = rank + 1
                u1listaddress[h, k1] = h
                axial[h, k1] = -1
            elseif m2 == n2
                u1listgrid[h, k1] = rank + 1
                u1listaddress[h, k1] = h
                axial[h, k1] = 1
            else
                St[m], St[n] = St[n], St[m]
                j2 = 1
                u = 0
                for i2 in 1 : N - 1
                    position = findfirst(isequal(N + 1 - i2), St)
                    k2 = findfirst(isequal(position), drop[i2][j2])
                    if k2 == nothing
                        u = -1
                        break
                    end
                    u += f[i2][j2][k2]
                    j2 = edge[i2][j2][k2]
                end
                if u != -1
                    u1 = u + 1
                    u1listgrid[h, k1] = searchsortedlast(grid, u1)
                    u1listaddress[h, k1] = u1 - grid[u1listgrid[h, k1]] + 1
                    axial[h, k1] = (n1 - m1) - (n2 - m2)
                end
                St[m], St[n] = St[n], St[m]
            end
        end
    end

    reverselist = zeros(Int32, chunk, N - 1)
    counts = zeros(Cint, Ncpu, N - 1)
    for k1 in 1 : N - 1
        jlist = [Int32[] for i1 in 1 : Ncpu]
        for j in 1 : chunk
            i1 = u1listgrid[j, k1]
            push!(jlist[i1], j)
        end
        reverselist[:, k1] .= vcat(jlist...)
        @. counts[:, k1] = length(jlist)
    end
    u1listgrid = Array{Int16}(undef, 0, 0)
    sendbuf1 = zeros(Int32, chunk)
    recvbuf1 = similar(sendbuf1)
    recvbuf1list = zeros(Int32, chunk, N - 1)
    for k1 in 1 : N - 1
        gatheringtest!(k1, reverselist, counts, u1listaddress, sendbuf1, recvbuf1)
        recvbuf1list[:, k1] .= recvbuf1
    end
    u1listaddress = Array{Int32}(undef, 0, 0)
    sendbuf1 = Int32[]
    recvbuf1 = Int32[]

    di = Vector{Int}[]
    for (i, c) in enumerate(color)
        push!(di, collect(Nc - i + 1 : Nc - i + c))
    end
    multiplicity = Int(prod(big.(vcat(di...))) * dim ÷ factorial(big(N)))
    if rank == 0
        println(multiplicity)
    end

    ketkm1 = zeros(Float64, chunk)
    Ψ = randn(Float64, chunk)
    ketk = copy(Ψ)
    ketk ./= sqrt(MPI.Allreduce(dot(ketk, ketk), +, comm))
    ketk1 = similar(ketk)
    z1 = similar(ketk)
    z2 = similar(ketk)
    sendbuf2 = similar(ketk)
    recvbuf2 = similar(sendbuf2)
    β = 0
    αlist = Float64[]
    βlist = Float64[]
    vals = Float64[]
    vecs = zeros(0, 0)
    vold = Inf
    k = 1
    while true
        H!(ketk1, ketk, z1, z2, NNsorted, reverselist, counts, axial, recvbuf1list, sendbuf2, recvbuf2)
        α = MPI.Allreduce(dot(ketk, ketk1), +, comm)
        push!(αlist, α)
        if k >= 2
            if rank == 0
                vals, vecs = LAPACK.stev!('V', copy(αlist), copy(βlist))
            end
            vals = MPI.bcast(vals, 0, comm)
            if abs((vals[2] - vold) / vals[2]) < 1e-14 || k == 100
                break
            end
            vold = vals[2]
        end
        @. ketk1 -= β * ketkm1 + α * ketk
        β = sqrt(MPI.Allreduce(dot(ketk1, ketk1), +, comm))
        if β == 0.0
            if rank == 0
                vals, vecs = LAPACK.stev!('V', copy(αlist), copy(βlist))
            end
            break
        end
        ketk1 ./= β
        ketkm1 .= ketk
        ketk .= ketk1
        push!(βlist, β)
        k += 1
    end
    vecs = MPI.bcast(vecs, 0, comm)
    if rank == 0
        println(vals[1 : min(10, length(vals))])
    end

    ketkm1 .= 0.0
    ketk .= Ψ
    β = 0.0
    Ψ .*= vecs[1, 1]
    for k in 1 : size(vecs, 1) - 1
        H!(ketk1, ketk, z1, z2, NNsorted, reverselist, counts, axial, recvbuf1list, sendbuf2, recvbuf2)
        α = αlist[k]
        axpy!(-β, ketkm1, ketk1)
        axpy!(-α, ketk, ketk1)
        β = βlist[k]
        ketk1 ./= β
        ketkm1 .= ketk
        ketk .= ketk1
        axpy!(vecs[k + 1, 1], ketk, Ψ)
    end
    Ψ ./= sqrt(MPI.Allreduce(dot(Ψ, Ψ), +, comm))

    PΨ = similar(Ψ)
    P = zeros(N, N)
    for i in 1 : N, j in 1 : N
        if i < j
            Pij!(i, j, PΨ, Ψ, z1, z2, reverselist, counts, axial, recvbuf1list, sendbuf2, recvbuf2)
            Pij = MPI.Reduce(dot(Ψ, PΨ), +, 0, comm)
            if rank == 0
                P[i, j] = Pij
                P[j, i] = Pij
            end
        end
    end
    if rank == 0
        println(P)
    end
end

MPI.Init()

comm = MPI.COMM_WORLD
Ncpu = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

id = parse(Int, ARGS[1])

if rank == 0
    println(id)
end

main(id, initialize())

MPI.Finalize()