# black box, uses score function estimator
function single_sample_gradient_estimate!(
        var_model::GenerativeFunction, var_model_args::Tuple,
        model::GenerativeFunction, model_args::Tuple, observations::ChoiceMap,
        scale_factor=1.)

    # sample from variational approximation
    var_trace = simulate(var_model, var_model_args)

    # compute learning signal
    constraints = merge(observations, get_choices(var_trace)) # TODO characterize what it means when not all var_model choices are in the model..
    (model_trace, model_log_weight) = generate(model, model_args, constraints)
    log_weight = model_log_weight - get_score(var_trace)

    # accumulate the weighted gradient
    accumulate_param_gradients!(var_trace, nothing, log_weight * scale_factor)

    # unbiased estimate of objective function, and trace
    (log_weight, var_trace, model_trace)
end

function vimco_geometric_baselines(log_weights)
    num_samples = length(log_weights)
    s = sum(log_weights)
    baselines = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        temp = log_weights[i]
        log_weights[i] = (s - log_weights[i]) / (num_samples - 1)
        baselines[i] = logsumexp(log_weights) - log(num_samples)
        log_weights[i] = temp
    end
    baselines
end

function logdiffexp(x, y)
    m = max(x, y)
    m + log(exp(x - m) - exp(y - m))
end

function vimco_arithmetic_baselines(log_weights)
    num_samples = length(log_weights)
    log_total_weight = logsumexp(log_weights)
    baselines = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        log_sum_f_without_i = logdiffexp(log_total_weight, log_weights[i])
        log_f_hat = log_sum_f_without_i - log(num_samples - 1)
        baselines[i] = logsumexp(log_sum_f_without_i, log_f_hat) - log(num_samples)
    end
    baselines
end

# black box, VIMCO gradient estimator
# for use in training models
function multi_sample_gradient_estimate!(
        var_model::GenerativeFunction, var_model_args::Tuple,
        model::GenerativeFunction, model_args::Tuple, observations::ChoiceMap,
        num_samples::Int, scale_factor=1., geometric=true)

    # sample from variational approximation multiple times
    traces = Vector{Any}(undef, num_samples)
    log_weights = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        traces[i] = simulate(var_model, var_model_args)
        constraints = merge(observations, get_choices(traces[i])) # TODO characterize as above
        model_weight, = assess(model, model_args, constraints)
        log_weights[i] = model_weight - get_score(traces[i])
    end

    # multi-sample log marginal likelihood estimate
    log_total_weight = logsumexp(log_weights)
    L = log_total_weight - log(num_samples)

    # baselines
    if geometric
        baselines = vimco_geometric_baselines(log_weights)
    else
        baselines = vimco_arithmetic_baselines(log_weights)
    end

    weights_normalized = exp.(log_weights .- log_total_weight)
    for i=1:num_samples
        learning_signal = (L - baselines[i]) - weights_normalized[i]
        accumulate_param_gradients!(traces[i], nothing, learning_signal * scale_factor)
    end

    # collection of traces and normalized importance weights, and estimate of
    # objective function
    (L, traces, weights_normalized)
end

function _maybe_accumulate_param_grad!(trace, optimizer::CompositeOptimizer, scale_factor::Real)
    return accumulate_param_gradients!(trace, nothing, scale_factor)
end

function _maybe_accumulate_param_grad!(trace, optimizer::Nothing, scale_factor::Real)
end

"""
    (elbo_estimate, traces, elbo_history) = black_box_vi!(
        model::GenerativeFunction, model_args::Tuple,
        [model_optimizer::CompositeOptimizer,]
        observations::ChoiceMap,
        var_model::GenerativeFunction, var_model_args::Tuple,
        var_model_optimizer::CompositeOptimizer;
        options...)

Fit the parameters of a variational model (`var_model`) to the posterior
distribution implied by the given `model` and `observations` using stochastic
gradient methods. Users may optionally specify a `model_optimizer` to jointly
update the parameters of `model`.

# Additional arguments:
- `iters=1000`: Number of iterations of gradient descent.
- `samples_per_iter=100`: Number of samples from the variational and generative
        model to accumulate gradients over before a single gradient step.
- `verbose=false`: If `true`, print information about the progress of fitting.
- `callback`: Callback function that takes `(iter, traces, elbo_estimate)`
        as input, where `iter` is the iteration number and `traces` are samples
        from `var_model` for that iteration.
"""
function black_box_vi!(
        model::GenerativeFunction, model_args::Tuple,
        model_optimizer::Union{CompositeOptimizer,Nothing},
        observations::ChoiceMap,
        var_model::GenerativeFunction, var_model_args::Tuple,
        var_model_optimizer::CompositeOptimizer;
        iters=1000, samples_per_iter=100, verbose=false,
        callback=(iter, traces, elbo_estimate) -> nothing)

    var_traces = Vector{Any}(undef, samples_per_iter)
    model_traces = Vector{Any}(undef, samples_per_iter)
    elbo_history = Vector{Float64}(undef, iters)
    for iter=1:iters

        # compute gradient estimate and objective function estimate
        elbo_estimate = 0.0
        # TODO multithread (note that this would require accumulate_param_gradients! to be threadsafe)
        for sample=1:samples_per_iter

            # accumulate the variational family gradients
            (log_weight, var_trace, model_trace) = single_sample_gradient_estimate!(
                var_model, var_model_args,
                model, model_args, observations, 1/samples_per_iter)
            elbo_estimate += (log_weight / samples_per_iter)

            # accumulate the generative model gradients
            _maybe_accumulate_param_grad!(model_trace, model_optimizer, 1.0 / samples_per_iter)

            # record the traces
            var_traces[sample] = var_trace
            model_traces[sample] = model_trace
        end
        elbo_history[iter] = elbo_estimate

        # print it
        verbose && println("iter $iter; est objective: $elbo_estimate")

        # callback
        callback(iter, var_traces, elbo_estimate)

        # update parameters of variational family
        apply_update!(var_model_optimizer)

        # update parameters of generative model
        if !isnothing(model_optimizer)
            apply_update!(model_optimizer)
        end
    end

    (elbo_history[end], var_traces, elbo_history, model_traces)
end

black_box_vi!(model::GenerativeFunction, model_args::Tuple,
              observations::ChoiceMap,
              var_model::GenerativeFunction, var_model_args::Tuple,
              var_model_optimizer::CompositeOptimizer; options...) =
    black_box_vi!(model, model_args, nothing, observations,
                  var_model, var_model_args, var_model_optimizer; options...)

"""
    (iwelbo_estimate, traces, iwelbo_history) = black_box_vimco!(
        model::GenerativeFunction, model_args::Tuple,
        [model_optimizer::CompositeOptimizer,]
        observations::ChoiceMap,
        var_model::GenerativeFunction, var_model_args::Tuple,
        var_model_optimizer::CompositeOptimizer,
        grad_est_samples::Int; options...)

Fit the parameters of a variational model (`var_model`) to the posterior
distribution implied by the given `model` and `observations` using stochastic
gradient methods applied to the [Variational Inference with Monte Carlo
Objectives](https://arxiv.org/abs/1602.06725) (VIMCO) lower bound on the
marginal likelihood. Users may optionally specify a `model_optimizer` to jointly
update the parameters of `model`.

# Additional arguments:
- `grad_est_samples::Int`: Number of samples for the VIMCO gradient estimate.
- `iters=1000`: Number of iterations of gradient descent.
- `samples_per_iter=100`: Number of samples from the variational and generative
        model to accumulate gradients over before a single gradient step.
- `geometric=true`: Whether to use the geometric or arithmetric baselines
        described in [Variational Inference with Monte Carlo
        Objectives](https://arxiv.org/abs/1602.06725)
- `verbose=false`: If `true`, print information about the progress of fitting.
- `callback`: Callback function that takes `(iter, traces, elbo_estimate)`
        as input, where `iter` is the iteration number and `traces` are samples
        from `var_model` for that iteration.
"""
function black_box_vimco!(
        model::GenerativeFunction, model_args::Tuple,
        model_optimizer::Union{CompositeOptimizer,Nothing}, observations::ChoiceMap,
        var_model::GenerativeFunction, var_model_args::Tuple,
        var_model_optimizer::CompositeOptimizer, grad_est_samples::Int;
        iters=1000, samples_per_iter=100, geometric=true, verbose=false,
        callback=(iter, traces, elbo_estimate) -> nothing)

    resampled_var_traces = Vector{Any}(undef, samples_per_iter)
    model_traces = Vector{Any}(undef, samples_per_iter)

    iwelbo_history = Vector{Float64}(undef, iters)
    for iter=1:iters

        # compute gradient estimate and objective function estimate
        iwelbo_estimate = 0.
        for sample=1:samples_per_iter

            # accumulate the variational family gradients
            (est, original_var_traces, weights) = multi_sample_gradient_estimate!(
                var_model, var_model_args,
                model, model_args, observations, grad_est_samples,
                1/samples_per_iter, geometric)
            iwelbo_estimate += (est / samples_per_iter)

            # record a variational trace obtained by resampling from the weighted collection
            resampled_var_traces[sample] = original_var_traces[categorical(weights)]

            # accumulate the generative model gradient estimator
            for (var_trace, weight) in zip(original_var_traces, weights)
                constraints = merge(observations, get_choices(var_trace))
                (model_trace, _) = generate(model, model_args, constraints)
                _maybe_accumulate_param_grad!(model_trace, model_optimizer, weight / samples_per_iter)
            end
        end
        iwelbo_history[iter] = iwelbo_estimate

        # print it
        verbose && println("iter $iter; est objective: $iwelbo_estimate")

        # callback
        callback(iter, resampled_var_traces, iwelbo_estimate)

        # update parameters of variational family
        apply_update!(var_model_optimizer)

        # update parameters of generative model
        if !isnothing(model_optimizer)
            apply_update!(model_optimizer)
        end

    end

    (iwelbo_history[end], resampled_var_traces, iwelbo_history, model_traces)
end

black_box_vimco!(model::GenerativeFunction, model_args::Tuple,
                 observations::ChoiceMap,
                 var_model::GenerativeFunction, var_model_args::Tuple,
                 var_model_optimizer::CompositeOptimizer,
                 grad_est_samples::Int; options...) =
    black_box_vimco!(model, model_args, nothing, observations,
                     var_model, var_model_args, var_model_optimizer,
                     grad_est_samples; options...)

export black_box_vi!, black_box_vimco!
