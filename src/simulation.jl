function sim_logit_gating(α, X;
    β=nothing, re_list=nothing, map_matrix=nothing)
    X = Array(X)
    gate = LogitGating(α, β)
    probs = exp.(LogitGatingEval(gate, X; re_list=re_list, map_matrix=map_matrix))
    return hcat([rand(Multinomial(1, probs[i, :])) for i in 1:size(X)[1]]...)'
end

function sim_dataset(α, X, model;
    β=nothing, re_list=nothing, map_matrix=nothing,
    exposure=nothing)
    X = Array(X)
    if isnothing(exposure)
        exposure = fill(1.0, size(X)[1])
    end
    model_expo = exposurize_model(model; exposure=exposure)
    gating_sim = sim_logit_gating(α, X; β=β, re_list=re_list, map_matrix=map_matrix)
    return vcat(
        [sim_components(model_expo[:, :, i]) * gating_sim[i, :] for i in 1:size(X)[1]]'...
    )
end