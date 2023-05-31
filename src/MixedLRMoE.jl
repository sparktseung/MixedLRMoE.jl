module MixedLRMoE

# Dependencies:
import Base.Threads: @threads, nthreads, threadid
import Base: @views
import LRMoE:
    @check_args,
    rowlogsumexp,
    params,
    _count_α,
    _count_params,
    loglik_exact,
    penalty_α,
    penalty_params,
    EM_M_expert_exact
import Distributions: MvNormal

end
