# Abstract type: AnyGating
abstract type AnyGating end

"""
    LogitGating(α, β)

Create a struct of coefficients for a gating function with α for the fixed effects, and β for the random effects.

# Arguments
- `α`: A matrix of coefficients for the fixed effects.
- `β`: A matrix of coefficients for the random effects. If β = nothing, then the gating function does not have random effects.

"""
struct LogitGating <: AnyGating
    α::Matrix{Float64}
    β::Union{Nothing,Matrix{Float64}}
    LogitGating(α) = new(α, nothing)
    LogitGating(α, β) = new(α, β)
end

Base.copy(s::LogitGating) = LogitGating(s.α, s.β)

"""
    LogitGatingEval(gate, x; re_list=nothing, map_matrix=nothing, check_args=true)

Evaluate the gating function at x and some specifications of random effects (if provided).

# Arguments
- `gate`: A `LogitGating` object.
- `x`: A matrix of covariates.

# Optional arguments
- `re_list`: A list of the realized values of of random effects.
- `map_matrix`: a matrix that maps each observation to their corresponding factor(s) in the random effect(s).
- `check_args`: If `true` (default), check the validity of the arguments.

# Return Values
- A matrix of the log probability values of the gating function.

"""
function LogitGatingEval(gate, x;
    re_list=nothing, map_matrix=nothing,
    check_args=true)
    if isnothing(re_list) || isnothing(map_matrix)
        check_args && @check_args(LogitGating, size(gate.α)[2] == size(x)[2])
        ax = x * gate.α'
        rowsum = rowlogsumexp(ax)
        return ax .- rowsum
    else
        w = mapRE(re_list, map_matrix)
        check_args && @check_args(LogitGating, size(gate.α)[1] == size(gate.β)[1])
        check_args && @check_args(LogitGating, size(gate.β)[2] == size(w)[2])
        ax = x * gate.α' + w * gate.β'
        rowsum = rowlogsumexp(ax)
        return ax .- rowsum
    end
end

"""
    LogitGatingSim(gate, x; re_μ_list=nothing, re_Σ_list=nothing, map_matrix=nothing, check_args=true)

Simulate the gating function at x and some specifications of random effects (if provided).

# Arguments
- `gate`: A `LogitGating` object.
- `x`: A matrix of covariates.

# Optional arguments
- `re_μ_list`: A list of the means of the random effects.
- `re_Σ_list`: A list of the diagonal covariance matrices of the random effects. By model assumption, 
    the random effects are assumed to be independent, so only a vector of the diagonal elements (i.e. the variances)
    is needed.
- `map_matrix`: a matrix that maps each observation to their corresponding factor(s) in the random effect(s).
- `check_args`: If `true` (default), check the validity of the arguments.

# Return Values
- A matrix of the log probability values of the gating function.

"""
function LogitGatingSim(gate, x;
    re_μ_list=nothing, re_Σ_list=nothing, map_matrix=nothing,
    check_args=true)
    if isnothing(re_μ_list) || isnothing(re_Σ_list) || isnothing(map_matrix)
        check_args && @check_args(LogitGating, size(gate.α)[2] == size(x)[2])
        ax = x * gate.α'
        rowsum = rowlogsumexp(ax)
        return ax .- rowsum
    else
        re_list_sample = [
            rand(MvNormal(μ, Σ)) for (μ, Σ) in zip(re_μ_list, re_Σ_list)
        ]
        w = mapRE(re_list_sample, map_matrix)
        check_args && @check_args(LogitGating, size(gate.α)[1] == size(gate.β)[1])
        check_args && @check_args(LogitGating, size(gate.β)[2] == size(w)[2])
        ax = x * gate.α' + w * gate.β'
        rowsum = rowlogsumexp(ax)
        return (ax .- rowsum)
    end
end

"""
    DiagMvNormal_KL(μ, Σ)

Calculate the KL divergence between a diagonal multivariate normal distribution and a standard normal distribution.

# Arguments
- `μ`: A vector of means.
- `Σ`: A vector of the diagonal elements in the covariance matrix.

"""
function DiagMvNormal_KL(μ, Σ)
    # return 0.5 * (-log(prod(Σ)) + sum(Σ) + sum(μ .* μ) - length(μ))
    return 0.5 * (-sum(log.(Σ)) + sum(Σ) + sum(μ .* μ) - length(μ))
end

function re_KL(re_μ_list, re_Σ_list)
    return sum([DiagMvNormal_KL(μ, Σ) for (μ, Σ) in zip(re_μ_list, re_Σ_list)])
end