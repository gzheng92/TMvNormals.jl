module TMvNormals
using Distributions, Statistics, LinearAlgebra
import Distributions: cdf, pdf, mean, cov, rand
using Parameters

include("mvnormal.jl")

"""
Truncated Multi-variate Normal distribution with lower limits [a] and upper limits [b].

To avoid limit issues, single- and un-truncated dimensions are handled as being truncated
at μ±10σ.  This shouldn't affect numerical results, as the code is also designed to error
out if the cdf routine is too inaccurate.
"""
struct TMvNormal <: ContinuousMultivariateDistribution
    μ::AbstractVector
    Σ::AbstractMatrix
    a::AbstractVector
    b::AbstractVector
    𝒩::AbstractMvNormal
    α::Float64
    K::Int
end

Base.show(io::IO, d::TMvNormal) =
    Distributions.show_multline(io, d, [(:𝒩, d.𝒩), (:lower, round.(d.a; digits=3)), (:upper, round.(d.b; digits=3))])

Base.broadcastable(d::TMvNormal) = Ref(d)
# function Base.length(::TMvNormal)
#     return 1
# end

# function Base.iterate(x::TMvNormal, state=1)
#     if state == 1
#         return (x, 2)
#     end
# end

"""
More convenient way to initialize the distribution with μ, Σ, a, and b
"""
function TMvNormal(μ::AbstractVector, Σ::AbstractMatrix, a::AbstractVector, b::AbstractVector)
    Σ = (Σ + Σ') / 2
    𝒩 = MvNormal(μ, Σ)
    a = ifelse.(
        isfinite.(a),
        a,
        μ - 10 * sqrt.(diag(Σ))
    )
    b = ifelse.(
        isfinite.(b),
        b,
        μ + 10 * sqrt.(diag(Σ))
    )
    return TMvNormal(
        μ,Σ,a,b,𝒩,
        cdf(𝒩, a, b),
        length(μ)
)
end

"""
More convenient way to initialize the distribution with 𝒩, a, and b
"""
function TMvNormal(𝒩::MvNormal, a::AbstractVector, b::AbstractVector)
    a = ifelse.(
        isfinite.(a),
        a,
        𝒩.μ - 10 * sqrt.(diag(𝒩.Σ))
    )
    b = ifelse.(
        isfinite.(b),
        b,
        𝒩.μ + 10 * sqrt.(diag(𝒩.Σ))
    )
    return TMvNormal(
        𝒩.μ, 𝒩.Σ, a, b, 𝒩,
        cdf(𝒩, a, b),
        length(𝒩.μ) 
    )
end

"""
Worst possible way to draw samples from the distribution
"""
function rand(d::TMvNormal, N::Int)
    @unpack a, b, K, 𝒩 = d
    x = Array{Float64}(undef, (N, K))
    for i = 1:N
        _trial = 0
        while true
            _trial += 1
            _x = rand(𝒩)
            if all(a .< _x .< b)
                x[i, :] = _x
                break
            elseif _trial > 100
                error("Failed to generate random sample")
        end
    end
    end
    return x
end

function cov2cor(Σ::AbstractMatrix)
    Σ = (Σ + Σ') / 2
    D = Diagonal((diag(Σ).^-0.5))
    return D * Σ * D
end

function cdf(𝒩::MvNormal, a::AbstractVector, b::AbstractVector)
    K = length(𝒩.μ)
    if K == 1
        𝒩 = Normal(𝒩.μ[1], √𝒩.Σ[1,1])
        return ifelse(b[1] == Inf, 1, cdf(𝒩, b[1])) - ifelse(a[1] == -Inf, 0, cdf(𝒩, a[1]))
    end
    
    if !all(a .<= b)
        error("a must be less than or equal to b")
    end
    
    val, err = qsimvnv(
        𝒩.Σ,
        a - 𝒩.μ,
        b - 𝒩.μ
    )
    if val == 0 || err < val / 100
        return val
    else
        error("Relative error exceeds 1%")
    end
end

function cdf(𝒩::MvNormal, b::AbstractVector)
    a = fill(-Inf, size(b))
    return cdf(𝒩, a, b)
end

function pdf(d::TMvNormal, x::AbstractVector)
    @unpack a, b, α, 𝒩 = d
    if all(a .<= x .<= b)
        return pdf(𝒩, x) / α
    else
        return 0
    end
end

"""
Calculate the marginal univariate density of a truncated multivariate normal.
Formula taken from Cartinhour (1990)
https://doi.org/10.1080/03610929008830197
"""
function pdf(d::TMvNormal, x::Number, dim::Number=1)
    @unpack a, b, K, α, μ, 𝒩, Σ = d
    idx = filter(!=(dim), 1:K)
    
    if !(a[dim] <= x <= b[dim])
        return 0
    end

    # get A̲₁, added a Symmetric call to handle small numerical errors
    Σ̲ = Symmetric(inv(inv(Matrix(Σ))[idx, idx]))
    a̲ = a[idx]
    b̲ = b[idx]
    σ̲ = Σ[idx, dim]
    # display(a)
    S(_x) = α^-1 * cdf(MvNormal(μ[idx] + (_x - μ[dim]) * σ̲ / Σ[dim,dim], Σ̲), a̲, b̲)
    return S(x) * pdf(Normal(μ[dim], √Σ[dim,dim]), x)
end

"""
Calculate the marginal univariate density of a truncated multivariate normal.
Formula taken from Manjunath Wilhelm (2021)
https://doi.org/10.35566/JBDS%2FV1N1%2FP2
"""
function pdf(d::TMvNormal, x::AbstractVector, margin::AbstractVector)
    @unpack α, K, μ, Σ, a, b = d
    margin = unique(margin)
    
    if length(margin) == 1
        return pdf(d, x[1], margin[1])
    elseif length(margin) != 2
        error("margin must be a vector of length 1 or 2, i got $(length(margin))")
    end
    
    q, r = margin
    
    if !all(a[[q, r]] .<= x .<= b[[q, r]])
        return 0
    end
    
    idx = filter(!∈(margin), 1:K)
    
    D = sqrt.(diag(Σ))
    R = cov2cor(Σ)

    # Manjunath Wilhelm (2021) says to use a z-transformed normal distribution for the marginal, but their code in `tmvtnorm.R` does not
    # Simulations agree with code, not paper
    ϕ = pdf(MvNormal(μ[[q, r]], Σ[[q, r], [q, r]]), x)
    
    # Multivariate regression coefficients
    β(s, q, r) = (R[s, q] - R[q, r] * R[s,r]) / (1 - R[q, r]^2)

    # Partial correlation coefficients
    function ρ(i, j, control::AbstractVector=[])
        if control == []
            return R[i,j]
        end
        (ρ(i, j, control[2:end]) - ρ(i, control[1], control[2:end]) * ρ(j, control[1], control[2:end])) / √((1 - ρ(j, control[1], control[2:end])^2) * (1 - ρ(i, control[1], control[2:end])^2))
    end    
    
    function ρ(i, j, control::Int)
        ρ(i, j, [control])
    end

    # ρ(s, q, r) = β(s, q, r) * √(1 - R[q, r]^2) / √(1 - R[s, r]^2)
    
    a = (a - μ) ./ D
    b = (b - μ) ./ D
    
    c = (x - μ[[q, r]]) ./ D[[q, r]]
    
    A(_q, _r, _s) = (a[_s] - β(_s, _q, _r) * c[1] - β(_s, _r, _q) * c[2]) / √((1 - R[_s, _q]^2) * (1 - ρ(_s, _r, _q)^2))
    B(_q, _r, _s) = (b[_s] - β(_s, _q, _r) * c[1] - β(_s, _r, _q) * c[2]) / √((1 - R[_s, _q]^2) * (1 - ρ(_s, _r, _q)^2))
    
    R₂ = Array{Float64}(undef, K, K)
    for i in idx,j in idx
        R₂[i,j] = ρ(i, j, [q,r])
    end
    R₂ = R₂[idx,idx]
    
    if K - 2 > 0
        Φᵈ⁻² = cdf(MvNormal(zeros(K - 2), R₂), A.(q, r, idx), B.(q, r, idx))
    else
        Φᵈ⁻² = 1
    end
    return α^-1 * ϕ * Φᵈ⁻²
end

function mean(d::TMvNormal, dim::Int)
    @unpack μ, K, Σ, a, b = d
    μ[dim] + reduce(
        +,
        Σ[1:K, dim] .* (pdf.(d, a, 1:K) - pdf.(d, b, 1:K))
    )
end

function mean(d::TMvNormal)
    @unpack K = d
    return mean.(d, 1:K)
end

function cov(d::TMvNormal, dims::AbstractVector{Int})
    
    if length(dims) != 2
        error("covariance matrix must be two dimensional")
    end
    
    @unpack K, μ, Σ, a, b = d
    
    D = sqrt.(diag(Σ))
    Σ
    a = (a - μ) # ./ D
    b = (b - μ) # ./ D
    d̂ = TMvNormal(zeros(K), Σ, a, b)
    μ̂ = mean(d̂)
    
    i, j = dims
    
    first_sum = reduce(
        +,
        map(
            k -> Σ[i,k] / Σ[k,k] * Σ[j,k] *
            (
                ifelse(isfinite(a[k]), a[k] * pdf(d̂, a[k], k), 0) -
                ifelse(isfinite(b[k]), b[k] * pdf(d̂, b[k], k), 0)
            ),
            1:K
        )
    )
    
    function inner_summand(q, k)
        pdf_term1 = pdf(d̂, [a[k], a[q]], [k, q]) - pdf(d̂, [a[k], b[q]], [k, q])
        pdf_term2 = pdf(d̂, [b[k], a[q]], [k, q]) - pdf(d̂, [b[k], b[q]], [k, q])
        (Σ[j, q] - Σ[k,q] * Σ[j,k] / Σ[k,k]) * (pdf_term1 - pdf_term2)
    end
    
    second_sum = reduce(
        +,
        map(
            k -> reduce(
                +,
                map(
                    q -> Σ[i, k] * inner_summand(q, k),
                    filter(!=(k), 1:K)
                )
            ),
            1:K
        )
    )
    
    return (Σ[i,j] + first_sum + second_sum - μ̂[i] * μ̂[j])
end

function cov(d::TMvNormal)
    @unpack K = d
    C = Array{Float64}(undef, K, K)
    for i = 1:K
        for j = i:K
            C[i,j] = cov(d, [i, j])
            C[j,i] = C[i,j]
    end
    end
    return Symmetric(C, :U)
end

"""
Test formulas using the 2 examples given in Manjunath Wilhelm (2021)
"""
function test_moments()
    println("2-d Example")
    let d = TMvNormal([0.5, 0.5], [1 1.2; 1.2 2], [-1, -Inf], [0.5, 1])
        ref_mean = [-0.152, -0.388]
        ref_cov = [0.163 0.161; 0.161 0.606]
        println("Computed mean:")
        display(round.(mean(d); digits=3))
        println("Reference mean:")
        display(round.(ref_mean; digits=3))
        println("Computed covariance:")
        display(round.(cov(d); digits=3))
        println("Reference covariance:")
        display(round.(ref_cov; digits=3))
    end
    
    println("3-d Example")
    let d = TMvNormal([0,0,0], [1.1 1.2 0; 1.2 2 -0.8; 0 -0.8 3], [-1, -Inf, -Inf], [0.5, Inf, Inf])
        ref_mean = [-0.210, -0.229, -0.0]
        ref_cov = [0.174 0.190 0.0; 0.190 0.898 -0.8; 0 -0.8 3.0]
        println("Computed mean:")
        display(round.(mean(d); digits=3))
        println("Reference mean:")
        display(round.(ref_mean; digits=3))
        println("Computed covariance:")
        display(round.(cov(d); digits=3))
        println("Reference covariance:")
        display(round.(ref_cov; digits=3))
    end
end

export TMvNormal, cov2cor, mean, cov, pdf, cdf
end
