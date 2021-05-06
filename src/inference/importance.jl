"""
    (traces, log_norm_weights, lml_est) = importance_sampling(model::GenerativeFunction,
        model_args::Tuple, observations::ChoiceMap, num_samples::Int, verbose=false)

    (traces, log_norm_weights, lml_est) = importance_sampling(model::GenerativeFunction,
        model_args::Tuple, observations::ChoiceMap,
        proposal::GenerativeFunction, proposal_args::Tuple,
        num_samples::Int, verbose=false)

Run importance sampling, returning a vector of traces with associated log weights.

The log-weights are normalized.
Also return the estimate of the marginal likelihood of the observations (`lml_est`).
The observations are addresses that must be sampled by the model in the given model arguments.
The first variant uses the internal proposal distribution of the model.
The second variant uses a custom proposal distribution defined by the given generative function.
All addresses of random choices sampled by the proposal should also be sampled by the model function.
Setting `verbose=true` prints a progress message every sample.
"""
function importance_sampling(model::GenerativeFunction{T,U}, model_args::Tuple,
                             observations::ChoiceMap,
                             num_samples::Int, verbose=false) where {T,U}
    traces = Vector{U}(undef, num_samples)
    log_weights = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        verbose && println("sample: $i of $num_samples")
        (traces[i], log_weights[i]) = generate(model, model_args, observations)
    end
    log_total_weight = logsumexp(log_weights)
    log_ml_estimate = log_total_weight - log(num_samples)
    log_normalized_weights = log_weights .- log_total_weight
    return (traces, log_normalized_weights, log_ml_estimate)
end

function importance_sampling(model::GenerativeFunction{T,U}, model_args::Tuple,
                             observations::ChoiceMap,
                             proposal::GenerativeFunction, proposal_args::Tuple,
                             num_samples::Int, verbose=false) where {T,U}
    traces = Vector{U}(undef, num_samples)
    log_weights = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        verbose && println("sample: $i of $num_samples")
        (proposed_choices, proposal_weight, _) = propose(proposal, proposal_args)
        constraints = merge(observations, proposed_choices)
        (traces[i], model_weight) = generate(model, model_args, constraints)
        log_weights[i] = model_weight - proposal_weight
    end
    log_total_weight = logsumexp(log_weights)
    log_ml_estimate = log_total_weight - log(num_samples)
    log_normalized_weights = log_weights .- log_total_weight
    return (traces, log_normalized_weights, log_ml_estimate)
end

"""
    (trace, lml_est) = importance_resampling(model::GenerativeFunction,
        model_args::Tuple, observations::ChoiceMap, num_samples::Int,
        verbose=false)

    (traces, lml_est) = importance_resampling(model::GenerativeFunction,
        model_args::Tuple, observations::ChoiceMap,
        proposal::GenerativeFunction, proposal_args::Tuple,
        num_samples::Int, verbose=false)

Run sampling importance resampling, returning a single trace.

Unlike `importance_sampling`, the memory used constant in the number of samples.

Setting `verbose=true` prints a progress message every sample.
"""
function importance_resampling(model::GenerativeFunction{T,U}, model_args::Tuple,
                               observations::ChoiceMap,
                               num_samples::Int; verbose=false)  where {T,U,V,W}
    (model_trace::U, log_weight) = generate(model, model_args, observations)
    log_total_weight = log_weight
    for i=2:num_samples
        verbose && println("sample: $i of $num_samples")
        (cand_model_trace, log_weight) = generate(model, model_args, observations)
        log_total_weight = logsumexp(log_total_weight, log_weight)
        if bernoulli(exp(log_weight - log_total_weight))
            model_trace = cand_model_trace
        end
    end
    log_ml_estimate = log_total_weight - log(num_samples)
    return (model_trace::U, log_ml_estimate::Float64)
end

function importance_resampling(model::GenerativeFunction{T,U}, model_args::Tuple,
                               observations::ChoiceMap,
                               proposal::GenerativeFunction{V,W}, proposal_args::Tuple,
                               num_samples::Int; verbose=false)  where {T,U,V,W}
    (proposal_choices, proposal_weight, _) = propose(proposal, proposal_args)
    constraints = merge(observations, proposal_choices)
    (model_trace::U, model_weight) = generate(model, model_args, constraints)
    log_total_weight = model_weight - proposal_weight
    for i=2:num_samples
        verbose && println("sample: $i of $num_samples")
        (proposal_choices, proposal_weight, _) = propose(proposal, proposal_args)
        constraints = merge(observations, proposal_choices)
        (cand_model_trace, model_weight) = generate(model, model_args, constraints)
        log_weight = model_weight - proposal_weight
        log_total_weight = logsumexp(log_total_weight, log_weight)
        if bernoulli(exp(log_weight - log_total_weight))
            model_trace = cand_model_trace
        end
    end
    log_ml_estimate = log_total_weight - log(num_samples)
    return (model_trace::U, log_ml_estimate::Float64)
end

"""
    log_ml_estimate = conditional_is_estimator(
        trace::Trace, observed::Selection, num_samples::Int)

Given a trace sampled from the conditional distribution given observed choices,
return an estimate of the log marginal likelihood of the observed choices that is a
stochastic upper bound on the true log marginal likelihood.
"""
function conditional_is_estimator(trace::Trace, observed::Selection, num_samples::Int)
    model = get_gen_fn(trace)
    model_args = get_args(trace)
    observations = get_selected(get_choices(trace), observed)
    log_weights = Vector{Float64}(undef, num_samples)
    log_weights[1] = project(trace, observed)
    for i=2:num_samples
        (_, log_weights[i]) = generate(model, model_args, observations)
    end
    log_total_weight = logsumexp(log_weights)
    log_ml_estimate = log_total_weight - log(num_samples)
    return log_ml_estimate
end

export importance_sampling, importance_resampling, conditional_is_estimator
