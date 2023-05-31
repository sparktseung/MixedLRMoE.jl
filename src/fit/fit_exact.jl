function _get_z_e_obs_threaded(
    gate, model, X, Y, exposure, re_μ_list, re_Σ_list, map_matrix, n_sims
)
    z_e_obs_mat = fill(0.0, size(X)[1], size(model)[2], nthreads())
    @threads for _ in 1:n_sims
        gate_em_eval = LogitGatingSim(
            gate,
            X;
            re_μ_list=re_μ_list,
            re_Σ_list=re_Σ_list,
            map_matrix=map_matrix,
            check_args=false,
        )
        ll_em_list = loglik_exact(Y, gate_em_eval, model; exposure=exposure)
        z_e_obs_mat[:, :, threadid()] .=
            exp.(ll_em_list.gate_expert_ll_comp .- ll_em_list.gate_expert_ll)
    end
    z_e_obs = sum(z_e_obs_mat; dims=3)[:, :, 1] ./ n_sims
    return z_e_obs
end

function fit_exact_VI(Y, X, α_init, model;
    exposure=nothing,
    β_init=nothing, map_matrix=nothing, re_μ_list=nothing, re_Σ_list=nothing, n_sims=10,
    penalty=true, pen_α=5.0, pen_params=nothing,
    ϵ=1e-03, α_iter_max=5.0, ecm_iter_max=200,
    print_steps=true)
    # Make variables accessible within the scope of `let`
    let α_em, β_em, gate_em, model_em, model_em_expo, ll_em_list, ll_em, ll_em_np,
        ll_em_old, ll_em_np_old, iter, z_e_obs, z_e_lat, k_e, params_old, re_μ_em, re_Σ_em
        # Initial loglik
        gate_init = LogitGating(α_init, β_init)
        ll_init_np_vec = fill(0.0, n_sims)
        @threads for sim in 1:n_sims
            gate_init_eval = LogitGatingSim(
                gate_init,
                X;
                re_μ_list=re_μ_list,
                re_Σ_list=re_Σ_list,
                map_matrix=map_matrix,
                check_args=false,
            )
            ll_np_list = loglik_exact(Y, gate_init_eval, model; exposure=exposure)
            ll_init_np_vec[sim] = ll_np_list.ll
        end
        ll_init_np = mean(ll_init_np_vec)
        ll_penalty = if penalty
            (penalty_α(gate_init.α, pen_α) + penalty_params(model, pen_params))
        else
            0.0
        end
        ll_re = if (isnothing(re_μ_list) || isnothing(re_Σ_list) || isnothing(map_matrix))
            0.0
        else
            -re_KL(re_μ_list, re_Σ_list)
        end
        ll_init = ll_init_np + ll_penalty + ll_re

        if print_steps
            @info("Initial loglik: $(ll_init_np) (no penalty), $(ll_init) (with penalty)")
        end

        # start em
        α_em = copy(α_init)
        β_em = copy(β_init)
        gate_em = LogitGating(α_em, β_em)
        re_μ_em = deepcopy(re_μ_list)
        re_Σ_em = deepcopy(re_Σ_list)
        model_em = copy(model)
        # model_em_expo = exposurize_model(model_em, exposure = exposure)
        ll_em_np = ll_init_np
        ll_em = ll_init
        ll_em_old = -Inf
        iter = 0

        ll_em_return = [ll_em]
        while iter < ecm_iter_max # (ll_em - ll_em_old > ϵ) & (iter < ecm_iter_max)
            # Update counter and loglikelihood
            iter = iter + 1
            ll_em_np_old = ll_em_np
            ll_em_old = ll_em

            # E-Step
            # z_e_obs = mean(z_e_obs_mat, dims=3)[:,:,1]
            gate_em = LogitGating(α_em, β_em)
            z_e_obs = _get_z_e_obs_threaded(
                gate_em, model_em, X, Y, exposure, re_μ_em, re_Σ_em, map_matrix, n_sims
            )
            z_e_lat = fill(1 / size(model)[2], size(X)[1], size(model)[2])
            k_e = fill(0.0, size(X)[1])

            # M-Step: α
            ll_em_temp = ll_em
            α_em, β_em, re_μ_em, re_Σ_em = EM_M_αβw_VI(X, gate_em.α, gate_em.β,
                z_e_obs, z_e_lat, k_e;
                map_matrix=map_matrix, re_μ_list=re_μ_em, re_Σ_list=re_Σ_em, n_sims=n_sims,
                αβ_iter_max=α_iter_max, w_iter_max=5,
                penalty=penalty, pen_α=pen_α)
            gate_em = LogitGating(α_em, β_em)
            ll_em_np_vec = fill(0.0, n_sims)
            @threads for sim in 1:n_sims
                gate_em_eval = LogitGatingSim(
                    gate_em,
                    X;
                    re_μ_list=re_μ_em,
                    re_Σ_list=re_Σ_em,
                    map_matrix=map_matrix,
                    check_args=false,
                )
                ll_em_list = loglik_exact(Y, gate_em_eval, model_em; exposure=exposure)
                ll_em_np_vec[sim] = ll_em_list.ll
            end
            ll_em_np = mean(ll_em_np_vec)
            ll_em_penalty = if penalty
                (penalty_α(gate_em.α, pen_α) + penalty_params(model_em, pen_params))
            else
                0.0
            end
            ll_re = if (isnothing(re_μ_em) || isnothing(re_Σ_em) || isnothing(map_matrix))
                0.0
            else
                -re_KL(re_μ_em, re_Σ_em)
            end
            ll_em = ll_em_np + ll_em_penalty + ll_re

            s = ll_em - ll_em_temp > 0 ? "+" : "-"
            pct = abs((ll_em - ll_em_temp) / ll_em_temp) * 100
            if print_steps
                @info(
                    "Iteration $(iter), updating α: $(ll_em_old) ->  $(ll_em), ( $(s) $(pct) % )"
                )
            end
            ll_em_temp = ll_em

            # M-Step: component distributions
            for j in 1:size(model)[1] # by dimension
                # for j in 1:1
                for k in 1:size(model)[2] # by component
                    params_old = params(model_em[j, k])

                    model_em[j, k] = EM_M_expert_exact(model_em[j, k],
                        Y[:, j], exposure,
                        # ll_em_list.expert_ll_pos_dim_comp[j][:,k],
                        # ll_em_list.expert_ll_dim_comp[j, k, :],
                        vec(z_e_obs[:, k]);
                        penalty=penalty, pen_pararms_jk=pen_params[j][k])

                    if print_steps
                        @info(
                            "Iteration $(iter), updating model[$j, $k]. Parameters:  $(params_old) ->  $(params(model_em[j,k]))"
                        )
                    end
                    # ll_em_temp = ll_em
                end
            end

            gate_em = LogitGating(α_em, β_em)
            # model_em_expo = exposurize_model(model_em, exposure = exposure)
            ll_em_np_vec = fill(0.0, n_sims)
            @threads for sim in 1:n_sims
                gate_em_eval = LogitGatingSim(
                    gate_em,
                    X;
                    re_μ_list=re_μ_em,
                    re_Σ_list=re_Σ_em,
                    map_matrix=map_matrix,
                    check_args=false,
                )
                ll_em_list = loglik_exact(Y, gate_em_eval, model_em; exposure=exposure)
                ll_em_np_vec[sim] = ll_em_list.ll
            end
            ll_em_np = mean(ll_em_np_vec)
            ll_em_penalty = if penalty
                (penalty_α(gate_em.α, pen_α) + penalty_params(model_em, pen_params))
            else
                0.0
            end
            ll_re = if (isnothing(re_μ_em) || isnothing(re_Σ_em) || isnothing(map_matrix))
                0.0
            else
                -re_KL(re_μ_em, re_Σ_em)
            end
            ll_em = ll_em_np + ll_em_penalty + ll_re

            push!(ll_em_return, ll_em)
        end

        converge = (ll_em - ll_em_old > ϵ) ? false : true
        AIC = -2.0 * ll_em_np + 2 * (_count_α(gate_em.α) + _count_params(model_em))
        BIC =
            -2.0 * ll_em_np +
            log(size(Y)[1]) * (_count_α(gate_em.α) + _count_params(model_em))

        return (α_fit=gate_em.α, β_fit=gate_em.β, model_fit=model_em, # re_fit = re_list_em,
            re_μ_fit=re_μ_em, re_Σ_fit=re_Σ_em,
            ll_history=ll_em_return,
            converge=converge, iter=iter,
            ll_np=ll_em_np, ll=ll_em,
            AIC=AIC, BIC=BIC)
    end
end