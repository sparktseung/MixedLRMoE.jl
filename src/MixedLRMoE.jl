module MixedLRMoE

# dependencies:
import Base.Threads: @threads, nthreads, threadid
import Base: @views
import InvertedIndices: Not
import LRMoE:
    @check_args,
    rowlogsumexp,
    params,
    exposurize_model,
    sim_components,
    EM_M_dQdα,
    EM_M_dQ2dα2
# _count_α,
# _count_params,
# loglik_exact,
# penalty_α,
# penalty_params,
# EM_M_expert_exact
import Distributions: MvNormal, Multinomial

# export

### source files
include("gating/logit.jl")
include("simulation.jl")
include("utils.jl")
include("fit/em.jl")

end
