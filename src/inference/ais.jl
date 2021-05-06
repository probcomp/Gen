"""
    (lml_est, trace, weights) = ais(
        model::GenerativeFunction, constraints::ChoiceMap,
        args_seq::Vector{Tuple}, argdiffs::Tuple,
        mcmc_kernel::Function)

Run annealed importance sampling, returning the log marginal likelihood estimate (`lml_est`).

The mcmc_kernel must satisfy detailed balance with respect to each step in the chain.
"""
function ais(
        model::GenerativeFunction, constraints::ChoiceMap,
        args_seq::Vector{<:Tuple}, argdiffs::Tuple, mcmc_kernel::Function)
    init_trace, init_weight = generate(model, args_seq[1], constraints)
    _ais(init_trace, init_weight, args_seq, argdiffs, mcmc_kernel)
end

function ais(
        trace::Trace, selection::Selection,
        args_seq::Vector{<:Tuple}, argdiffs::Tuple, mcmc_kernel::Function)
    init_trace, = update(init_trace, args_seq[1], argdiffs, EmptyChoiceMap())
    init_weight = project(trace, ComplementSelection(selection))
    _ais(init_trace, init_weight, args_seq, argdiffs, mcmc_kernel)
end

function _ais(
        trace::Trace, init_weight::Float64, args_seq::Vector{<:Tuple},
        argdiffs::Tuple, mcmc_kernel::Function)
    @assert get_args(trace) == args_seq[1]

    # run forward AIS
    weights = Float64[]
    lml_est = init_weight
    push!(weights, init_weight)
    for intermediate_args in args_seq[2:end-1]
        trace = mcmc_kernel(trace)
        (trace, weight, _, discard) = update(trace, intermediate_args, argdiffs, EmptyChoiceMap())
        if !isempty(discard)
            error("Change to arguments cannot cause random choices to be removed from trace")
        end
        lml_est += weight
        push!(weights, weight)
    end
    trace = mcmc_kernel(trace)
    (trace, weight, _, discard) = update(
        trace, args_seq[end], argdiffs, EmptyChoiceMap())
        if !isempty(discard)
            error("Change to arguments cannot cause random choices to be removed from trace")
        end
    lml_est += weight
    push!(weights, weight)

    # do MCMC at the very end
    trace = mcmc_kernel(trace)

    return (lml_est, trace, weights)
end

"""
    (lml_est, weights) = reverse_ais(
        model::GenerativeFunction, constraints::ChoiceMap,
        args_seq::Vector{Tuple}, argdiffs::Tuple,
        mcmc_kernel::Function)

Run reverse annealed importance sampling, returning the log marginal likelihood estimate (`lml_est`).

`constraints` must be a choice map that uniquely determines a trace of the model for the final arguments in the argument sequence.
The mcmc_kernel must satisfy detailed balance with respect to each step in the chain.
"""
function reverse_ais(
        model::GenerativeFunction, constraints::ChoiceMap,
        args_seq::Vector, argdiffs::Tuple,
        mh_rev::Function, output_addrs::Selection; safe=true)

    # construct final model trace from the inferred choices and all the fixed choices
    (trace, should_be_score) = generate(model, args_seq[end], constraints)
    init_score = get_score(trace)
    if safe && !isapprox(should_be_score, init_score) # check it's deterministic
        error("Some random choices may have been unconstrained")
    end

    # do mh at the very beginning
    trace = mh_rev(trace)

    # run backward AIS
    lml_est = 0.
    weights = Float64[]
    for model_args in reverse(args_seq[1:end-1])
        (trace, weight, _, _) = update(trace, model_args, argdiffs, EmptyChoiceMap())
        safe && isnan(weight) && error("NaN weight")
        lml_est -= weight
        push!(weights, -weight)
        trace = mh_rev(trace)
    end

    # get pi_1(z_0) / q(z_0) -- the weight that would be returned by the initial 'generate' call
    # select the addresses that would be constrained by the call to generate inside to AIS.simulate()
    @assert get_args(trace) == args_seq[1]
    #score_from_project = project(trace, ComplementSelection(output_addrs))
    score_from_project = project(trace, output_addrs)
    lml_est += score_from_project
    push!(weights, score_from_project)
    if isnan(score_from_project)
        error("NaN score_from_project")
    end

    return (lml_est, reverse(weights))
end

export ais, reverse_ais
