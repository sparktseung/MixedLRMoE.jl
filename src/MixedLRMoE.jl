module MixedLRMoE

# dependencies:
import Base.Threads: @threads, nthreads, threadid
import Base: @views
import Distributions: MvNormal, Multinomial
import InvertedIndices: Not
import LinearAlgebra: I
import Statistics: mean
import LRMoE:
    @check_args,
    rowlogsumexp,
    params,
    exposurize_model,
    sim_components,
    EM_M_dQdα,
    EM_M_dQ2dα2,
    loglik_exact,
    penalty_α,
    penalty_params,
    EM_M_expert_exact,
    _count_α,
    _count_params

export
    fit_exact_VI

### source files
include("gating/logit.jl")
include("simulation.jl")
include("utils.jl")
include("fit/em.jl")
include("fit/fit_exact.jl")

end
