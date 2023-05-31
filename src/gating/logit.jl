# Abstract type: AnyGating
abstract type AnyGating end

struct LogitGating <: AnyGating
    α::Matrix{Float64}
    β::Union{Nothing,Matrix{Float64}}
    LogitGating(α) = new(α, nothing)
    LogitGating(α, β) = new(α, β)
end

Base.copy(s::LogitGating) = LogitGating(s.α, s.β)

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

function DiagMvNormal_KL(μ, Σ)
    # return 0.5 * (-log(prod(Σ)) + sum(Σ) + sum(μ .* μ) - length(μ))
    return 0.5 * (-sum(log.(Σ)) + sum(Σ) + sum(μ .* μ) - length(μ))
end

function re_KL(re_μ_list, re_Σ_list)
    return sum([DiagMvNormal_KL(μ, Σ) for (μ, Σ) in zip(re_μ_list, re_Σ_list)])
end