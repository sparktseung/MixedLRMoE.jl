function _EM_M_dQdβj(w, comp_zkz_j, comp_zkz_marg, pp_j)
    return vec(sum(w .* (comp_zkz_j - comp_zkz_marg .* pp_j); dims=1))
end

function _EM_M_dQ2dβj2(w, comp_zkz_marg, pp_j, qq_j)
    return -w' * (comp_zkz_marg .* pp_j .* qq_j .* w)
end

function _EM_M_dQdwl(wl, βl, map_vec, comp_zkz, comp_zkz_marg, pp)
    βl_T = reshape(βl, (1, length(βl)))
    sums_j = sum((comp_zkz .- comp_zkz_marg .* pp) .* βl_T; dims=2)
    # n*1       #  n*g              n*1       n*g     1*g
    result = fill(NaN, length(wl))
    for l in 1:length(wl)
        result[l] = sum(sums_j[map_vec .== l])
    end
    return result
end

function _EM_M_dQ2dwl2(wl, βl, map_vec, comp_zkz_marg, pp)
    βl_T = reshape(βl, (1, length(βl)))
    β_weighted = sum(pp .* βl_T; dims=2)
    β2_weighted = sum(pp .* (βl_T .* βl_T); dims=2)
    sums_j = sum(comp_zkz_marg .* (-β2_weighted .+ β_weighted .* β_weighted); dims=2)

    result = fill(NaN, length(wl))
    for l in 1:length(wl)
        result[l] = sum(sums_j[map_vec .== l])
    end
    return result
end

function _dQdα_threaded(
    re_list_sample, map_matrix, X, α, β, comp_zkz, j, comp_zkz_marg, penalty, pen_α
)
    w = mapRE(re_list_sample, map_matrix)

    gate_body = X * α' + w * β'
    pp = exp.(gate_body .- rowlogsumexp(gate_body))
    qqj = exp.(rowlogsumexp(gate_body[:, Not(j)]) - rowlogsumexp(gate_body))

    dαj = (
        EM_M_dQdα(X, comp_zkz[:, j], comp_zkz_marg, pp[:, j]) .-
        (penalty ? vec(α[j, :] ./ pen_α^2) : 0.0)
    )
    dαjdαj = (
        EM_M_dQ2dα2(X, comp_zkz_marg, pp[:, j], qqj) -
        (penalty ? (1.0 ./ pen_α^2) * I(size(α)[2]) : (1e-07) * I(size(α)[2]))
    )

    return dαj, dαjdαj
end

function _dQdα_all_samples(
    re_list_sample_list, map_matrix, X, α, β, comp_zkz, j, comp_zkz_marg, penalty, pen_α
)
    n_sims = length(re_list_sample_list)
    αj_len = length(α[j, :])

    dαj_mat = zeros(Float64, αj_len, n_sims)
    dαjdαj_mat = zeros(Float64, αj_len, αj_len, n_sims)

    @threads for sample_no in 1:n_sims
        dαj_mat[:, sample_no], dαjdαj_mat[:, :, sample_no] = _dQdα_threaded(
            re_list_sample_list[sample_no],
            map_matrix,
            X,
            α,
            β,
            comp_zkz,
            j,
            comp_zkz_marg,
            penalty,
            pen_α,
        )
    end

    dαj = sum(dαj_mat; dims=2)[:, 1] ./ n_sims
    dαjdαj = sum(dαjdαj_mat; dims=3)[:, :, 1] ./ n_sims

    return dαj, dαjdαj
end

function _dQdβ_threaded(
    re_list_sample, map_matrix, X, α, β, comp_zkz, j, comp_zkz_marg, penalty, pen_α
)
    w = mapRE(re_list_sample, map_matrix)

    gate_body = X * α' + w * β'
    pp = exp.(gate_body .- rowlogsumexp(gate_body))
    qqj = exp.(rowlogsumexp(gate_body[:, Not(j)]) - rowlogsumexp(gate_body))

    dβj = (
        _EM_M_dQdβj(w, comp_zkz[:, j], comp_zkz_marg, pp[:, j]) .-
        (penalty ? vec(β[j, :] ./ pen_α^2) : 0.0)
    )
    dβjdβj = (
        _EM_M_dQ2dβj2(w, comp_zkz_marg, pp[:, j], qqj) -
        (penalty ? (1.0 ./ pen_α^2) * I(size(β)[2]) : (1e-07) * I(size(β)[2]))
    )

    return dβj, dβjdβj
end

function _dQdβ_all_samples(
    re_list_sample_list, map_matrix, X, α, β, comp_zkz, j, comp_zkz_marg, penalty, pen_α
)
    n_sims = length(re_list_sample_list)
    βj_len = length(β[j, :])

    dβj_mat = zeros(Float64, βj_len, n_sims)
    dβjdβj_mat = zeros(Float64, βj_len, βj_len, n_sims)

    @threads for sample_no in 1:n_sims
        dβj_mat[:, sample_no], dβjdβj_mat[:, :, sample_no] = _dQdβ_threaded(
            re_list_sample_list[sample_no], map_matrix, X, α, β, comp_zkz, j, comp_zkz_marg,
            penalty, pen_α)
    end

    dβj = sum(dβj_mat; dims=2)[:, 1] ./ n_sims
    dβjdβj = sum(dβjdβj_mat; dims=3)[:, :, 1] ./ n_sims

    return dβj, dβjdβj
end

function _dQdμldΣl_thread(
    re_list_sample,
    map_matrix,
    X,
    α,
    β,
    comp_zkz,
    comp_zkz_marg,
    re_μ_list,
    re_Σ_list,
    l,
    sqrtΣl,
)
    w = mapRE(re_list_sample, map_matrix)

    # Recover standard normal sample
    vl = (re_list_sample[l] .- re_μ_list[l]) ./ sqrtΣl

    gate_body = X * α' + w * β'
    pp = exp.(gate_body .- rowlogsumexp(gate_body))

    # dwl and dwldwl
    dwl = _EM_M_dQdwl(
        re_list_sample[l], β[:, l], map_matrix[:, l], comp_zkz, comp_zkz_marg, pp
    )
    dwldwl = _EM_M_dQ2dwl2(re_list_sample[l], β[:, l], map_matrix[:, l], comp_zkz_marg, pp)

    # dμl and dμldμl
    dμl = dwl - re_μ_list[l]
    dμldμl = dwldwl - fill(1.0, length(re_μ_list[l]))

    # dΣl and dΣldΣl
    dΣl = dwl ./ vl + 1 ./ sqrtΣl - sqrtΣl
    dΣldΣl = dwldwl ./ (vl) .^ 2 - 1 ./ (sqrtΣl) .^ 2 - fill(1.0, length(re_μ_list[l]))

    return dμl, dμldμl, dΣl, dΣldΣl
end

function _dQdμldΣl_all_samples(
    re_list_sample_list,
    map_matrix,
    X,
    α,
    β,
    comp_zkz,
    comp_zkz_marg,
    re_μ_list,
    re_Σ_list,
    l,
    sqrtΣl,
)
    n_sims = length(re_list_sample_list)
    μl_len = length(re_μ_list[l])

    dμl_mat = zeros(Float64, μl_len, n_sims)
    dμldμl_mat = zeros(Float64, μl_len, n_sims)
    dsqrtΣl_mat = zeros(Float64, μl_len, n_sims)
    dsqrtΣldsqrtΣl_mat = zeros(Float64, μl_len, n_sims)

    @threads for sample_no in 1:n_sims
        (dμl_mat[:, sample_no],
        dμldμl_mat[:, sample_no],
        dsqrtΣl_mat[:, sample_no],
        dsqrtΣldsqrtΣl_mat[:, sample_no]) = _dQdμldΣl_thread(
            re_list_sample_list[sample_no],
            map_matrix,
            X,
            α,
            β,
            comp_zkz,
            comp_zkz_marg,
            re_μ_list,
            re_Σ_list,
            l,
            sqrtΣl,
        )
    end

    dμl = sum(dμl_mat; dims=2)[:, 1] ./ n_sims
    dμldμl = sum(dμldμl_mat; dims=2)[:, 1] ./ n_sims

    dsqrtΣl = sum(dsqrtΣl_mat; dims=2)[:, 1] ./ n_sims
    dsqrtΣldsqrtΣl = sum(dsqrtΣldsqrtΣl_mat; dims=2)[:, 1] ./ n_sims

    return dμl, dμldμl, dsqrtΣl, dsqrtΣldsqrtΣl
end

function EM_M_αβw_VI(X, α, β, z_e_obs, z_e_lat, k_e;
    map_matrix=nothing, re_μ_list=nothing, re_Σ_list=nothing, n_sims=10,
    αβ_iter_max=5, w_iter_max=5,
    penalty=true, pen_α=5)
    let comp_zkz, comp_zkz_marg, α_new, α_old, β_new, β_old, iter, gate_new, w
        comp_zkz = z_e_obs .+ (k_e .* z_e_lat)
        comp_zkz_marg = vec(sum(comp_zkz; dims=2))

        α_new = copy(α)
        α_old = copy(α_new) .- Inf
        β_new = copy(β)
        β_old = copy(β_new) .- Inf

        ll_old = -Inf
        iter = fill(0, size(α_new)[1])

        re_list_sample_list = [[
            rand(MvNormal(μ, Σ)) for (μ, Σ) in zip(re_μ_list, re_Σ_list)
        ]]
        for _ in 1:(n_sims - 1)
            re_list_sample = [
                rand(MvNormal(μ, Σ)) for (μ, Σ) in zip(re_μ_list, re_Σ_list)
            ]
            push!(re_list_sample_list, re_list_sample)
        end

        for j in 1:(size(α)[1] - 1)
            while (iter[j] < αβ_iter_max) & (sum((α_new[j, :] - α_old[j, :]) .^ 2) > 1e-08)
                α_old[j, :] = α_new[j, :]

                dαj, dαjdαj = _dQdα_all_samples(
                    re_list_sample_list,
                    map_matrix,
                    X,
                    α_new,
                    β_new,
                    comp_zkz,
                    j,
                    comp_zkz_marg,
                    penalty,
                    pen_α,
                )

                α_new[j, :] = α_new[j, :] .- inv(dαjdαj) * dαj

                if isnan(sum(α_new[j, :])) || isinf(sum(α_new[j, :])) # Prevent error
                    α_new[j, :] = α_old[j, :]
                end

                iter[j] = iter[j] + 1
            end
        end

        if size(α)[1] > 2 # only update β when there are >= 3 classes
            iter = fill(0, size(β_new)[1])

            for j in 2:(size(α)[1] - 1)
                while (iter[j] < αβ_iter_max) &
                      (sum((β_new[j, :] - β_old[j, :]) .^ 2) > 1e-08)
                    β_old[j, :] = β_new[j, :]

                    dβj, dβjdβj = _dQdβ_all_samples(
                        re_list_sample_list,
                        map_matrix,
                        X,
                        α_new,
                        β_new,
                        comp_zkz,
                        j,
                        comp_zkz_marg,
                        penalty,
                        pen_α,
                    )

                    β_new[j, :] = β_new[j, :] .- inv(dβjdβj) * dβj

                    if isnan(sum(β_new[j, :])) || isinf(sum(β_new[j, :])) # Prevent error
                        β_new[j, :] = β_old[j, :]
                    end

                    iter[j] = iter[j] + 1
                end
            end
        end

        re_μ_new = deepcopy(re_μ_list)
        re_Σ_new = deepcopy(re_Σ_list)

        iter = fill(0, length(re_μ_list))
        for l in 1:length(re_μ_list)
            while (iter[l] < 1)
                re_μ_old = copy(re_μ_new[l])
                re_Σ_old = copy(re_Σ_new[l])

                sqrtΣl = sqrt.(re_Σ_new[l])

                dμl, dμldμl, dsqrtΣl, dsqrtΣldsqrtΣl = _dQdμldΣl_all_samples(
                    re_list_sample_list,
                    map_matrix,
                    X,
                    α_new,
                    β_new,
                    comp_zkz,
                    comp_zkz_marg,
                    re_μ_new,
                    re_Σ_new,
                    l,
                    sqrtΣl,
                )

                # Update μ
                re_μ_new[l] .= re_μ_new[l] .- dμl ./ dμldμl
                # Update Σ
                sqrtΣl .= sqrtΣl .- dsqrtΣl ./ dsqrtΣldsqrtΣl
                # sqrtΣl[sqrtΣl .<= 1e-8] .= 1e-8 # to prevent numerical underflow
                re_Σ_new[l] .= (sqrtΣl) .^ 2

                if isnan(sum(re_μ_new[l])) || isinf(sum(re_μ_new[l])) # Prevent error
                    re_μ_new[l] .= re_μ_old
                end

                if isnan(sum(re_Σ_new[l])) || isinf(sum(re_Σ_new[l])) # Prevent error
                    re_Σ_new[l] .= re_Σ_old
                end

                iter[l] = iter[l] + 1
            end
        end

        # GC.gc(true)

        return α_new, β_new, re_μ_new, re_Σ_new
    end
end