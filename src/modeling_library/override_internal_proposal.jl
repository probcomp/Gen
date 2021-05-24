
# Generative function combiantor that overrides internal proposal with another
# generative function

# Not yet implemented:
# - update
# - project
# - choice_gradients
# - accumulate_param_gradients!

struct ReplaceProposalGFTrace{U} <: Trace
    model_trace::U
    gen_fn::GenerativeFunction
end

get_args(tr::ReplaceProposalGFTrace) = get_args(tr.model_trace)
get_retval(tr::ReplaceProposalGFTrace) = get_retval(tr.model_trace)
get_choices(tr::ReplaceProposalGFTrace) = get_choices(tr.model_trace)
get_score(tr::ReplaceProposalGFTrace) = get_score(tr.model_trace)

struct ReplaceProposalGF{T,U} <: GenerativeFunction{T,ReplaceProposalGFTrace{U}}
    model::GenerativeFunction{T,U}
    proposal::GenerativeFunction
end

get_gen_fn(tr::ReplaceProposalGFTrace) = tr.gen_fn

# gradient ops not implemented yet
has_argument_grads(f::ReplaceProposalGF) = map(_->false,has_argument_grads(f.model))
accepts_output_grad(f::ReplaceProposalGF) = false

function project(tr::ReplaceProposalGFTrace, ::EmptySelection)
    return project(tr.model_trace, EmptySelection())
end

function simulate(gen_fn::ReplaceProposalGF, args::Tuple)
    tr = simulate(gen_fn.model, args)
    return ReplaceProposalGFTrace(tr, gen_fn)
end

function generate(gen_fn::ReplaceProposalGF, args::Tuple, constraints::ChoiceMap)
    (proposed_choices, proposal_weight, _) = propose(gen_fn.proposal, (constraints, args...))
    all_constraints = merge(proposed_choices, constraints)
    new_tr, model_weight = generate(gen_fn.model, args, all_constraints)
    @assert isapprox(model_weight, get_score(new_tr))
    weight = model_weight - proposal_weight
    return (ReplaceProposalGFTrace(new_tr, gen_fn), weight)
end

function regenerate(trace::ReplaceProposalGFTrace, args::Tuple, argdiffs::Tuple, selection::Selection)
    gen_fn = get_gen_fn(trace)
    prev_args = get_args(trace)

    # u <- create choice map u containing addresses from trace, except for those in selection
    u = get_selected(get_choices(trace), complement(selection))

    # then, run generate with that u to obtain new-trace t', and weight w = p(t'; x') / q(t; x, u')
    (new_trace, p_weight) = generate(gen_fn, args, u)

    # then, create choice map u' containing addresses from new-trace, except for those in selection
    u_backward = get_selected(get_choices(new_trace), complement(selection))

    # then, run generate on custom_q to obtain q(t; x, u')
    (_, q_weight) = generate(gen_fn.proposal, (u_backward, prev_args...), get_choices(trace)) # NOTE there will be extra choices
    
    # then, use get_score(trace) and subtracct it from the weight
    weight = p_weight + q_weight - get_score(trace)

    return (new_trace, weight, UnknownChange())
end

function override_internal_proposal(p, q)
    return ReplaceProposalGF(p, q)
end

export override_internal_proposal
