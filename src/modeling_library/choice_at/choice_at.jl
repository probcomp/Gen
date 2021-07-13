# TODO optimize ChoiceAtTrace using type parameters

struct ChoiceAtTrace <: Trace
    gen_fn::GenerativeFunction # the ChoiceAtCombinator (not the kernel)
    value::Any
    key::Any
    kernel_args::Tuple
    score::Float64
end

get_args(trace::ChoiceAtTrace) = (trace.kernel_args..., trace.key)
get_retval(trace::ChoiceAtTrace) = trace.value
get_score(trace::ChoiceAtTrace) = trace.score
get_gen_fn(trace::ChoiceAtTrace) = trace.gen_fn

struct ChoiceAtChoiceMap{T,K} <: ChoiceMap
    key::K
    value::T
end

get_choices(trace::ChoiceAtTrace) = ChoiceAtChoiceMap(trace.key, trace.value)
Base.isempty(::ChoiceAtChoiceMap) = false
function get_address_schema(::Type{T}) where {T<:ChoiceAtChoiceMap}
    SingleDynamicKeyAddressSchema()
end
get_value(choices::ChoiceAtChoiceMap, addr::Pair) = _get_value(choices, addr)
has_value(choices::ChoiceAtChoiceMap, addr::Pair) = _has_value(choices, addr)
has_value(choices::ChoiceAtChoiceMap, addr) = addr == choices.key
function get_value(choices::ChoiceAtChoiceMap{T,K}, addr::K) where {T,K}
    choices.key == addr ? choices.value : throw(KeyError(choices, addr))
end
get_submaps_shallow(choices::ChoiceAtChoiceMap) = ()
get_values_shallow(choices::ChoiceAtChoiceMap) = ((choices.key, choices.value),)

struct ChoiceAtCombinator{T,K} <: GenerativeFunction{T, ChoiceAtTrace}
    dist::Distribution{T}
end

accepts_output_grad(gen_fn::ChoiceAtCombinator) = has_output_grad(gen_fn.dist)

# TODO
# accepts_output_grad is true if the return value is dependent on the 'gradient source elements'
# if the random choice itself is not a 'gradient source element' then it is independent (false)
# if the random choice is a 'gradient source element', then the return value is dependent (true)
# we will consider the random choice as a gradient source element if the
# distribution has has_output_grad = true)

function choice_at(dist::Distribution{T}, ::Type{K}) where {T,K}
    ChoiceAtCombinator{T,K}(dist)
end

unpack_choice_at_args(args) = (args[end], args[1:end-1])

function assess(gen_fn::ChoiceAtCombinator{T,K}, args::Tuple, choices::ChoiceMap) where {T,K}
    local key::K
    local value::T
    (key, kernel_args) = unpack_choice_at_args(args)
    value = get_value(choices, key)
    weight = logpdf(gen_fn.dist, value, kernel_args...)
    (weight, value)
end

function propose(gen_fn::ChoiceAtCombinator{T,K}, args::Tuple) where {T,K}
    local key::K
    local value::T
    (key, kernel_args) = unpack_choice_at_args(args)
    value = random(gen_fn.dist, kernel_args...)
    score = logpdf(gen_fn.dist, value, kernel_args...)
    choices = ChoiceAtChoiceMap(key, value)
    (choices, score, value)
end

function simulate(gen_fn::ChoiceAtCombinator, args::Tuple)
    (key, kernel_args) = unpack_choice_at_args(args)
    value = random(gen_fn.dist, kernel_args...)
    score = logpdf(gen_fn.dist, value, kernel_args...)
    ChoiceAtTrace(gen_fn, value, key, kernel_args, score)
end

function generate(gen_fn::ChoiceAtCombinator{T,K}, args::Tuple, choices::ChoiceMap) where {T,K}
    local key::K
    local value::T
    (key, kernel_args) = unpack_choice_at_args(args)
    constrained = has_value(choices, key)
    value = constrained ? get_value(choices, key) : random(gen_fn.dist, kernel_args...)
    score = logpdf(gen_fn.dist, value, kernel_args...)
    trace = ChoiceAtTrace(gen_fn, value, key, kernel_args, score)
    weight = constrained ? score : 0.
    (trace, weight)
end

function project(trace::ChoiceAtTrace, selection::Selection)
    (trace.key in selection) ? trace.score : 0.
end

function update(trace::ChoiceAtTrace, args::Tuple, argdiffs::Tuple,
                choices::ChoiceMap)
    (key, kernel_args) = unpack_choice_at_args(args)
    key_changed = (key != trace.key)
    constrained = has_value(choices, key)
    if key_changed && constrained
        new_value = get_value(choices, key)
        discard = ChoiceAtChoiceMap(trace.key, trace.value)
    elseif !key_changed && constrained
        new_value = get_value(choices, key)
        discard = ChoiceAtChoiceMap(key, trace.value)
    elseif !key_changed && !constrained
        new_value = trace.value
        discard = EmptyChoiceMap()
    else
        error("New address $key not constrained in update")
    end
    new_score = logpdf(trace.gen_fn.dist, new_value, kernel_args...)
    new_trace = ChoiceAtTrace(trace.gen_fn, new_value, key, kernel_args, new_score)
    weight = new_score - trace.score
    (new_trace, weight, UnknownChange(), discard)
end

function regenerate(trace::ChoiceAtTrace, args::Tuple, argdiffs::Tuple,
                    selection::Selection)
    (key, kernel_args) = unpack_choice_at_args(args)
    key_changed = (key != trace.key)
    selected = key in selection
    if !key_changed && selected
        new_value = random(trace.gen_fn.dist, kernel_args...)
    elseif !key_changed && !selected
        new_value = trace.value
    elseif key_changed && !selected
        new_value = random(trace.gen_fn.dist, kernel_args...)
    else
        error("Cannot select new address $key in regenerate")
    end
    new_score = logpdf(trace.gen_fn.dist, new_value, kernel_args...)
    if !key_changed && selected
        weight = 0.
    elseif !key_changed && !selected
        weight = new_score - trace.score
    elseif key_changed && !selected
        weight = 0.
    end
    new_trace = ChoiceAtTrace(trace.gen_fn, new_value, key, kernel_args, new_score)
    (new_trace, weight, UnknownChange())
end

function choice_gradients(trace::ChoiceAtTrace, selection::Selection, retval_grad)
    if retval_grad != nothing && !has_output_grad(trace.gen_fn.dist)
        error("return value gradient not accepted but one was provided")
    end
    kernel_arg_grads = logpdf_grad(trace.gen_fn.dist, trace.value, trace.kernel_args...)
    if trace.key in selection
        value_choices = ChoiceAtChoiceMap(trace.key, trace.value)
        choice_grad = kernel_arg_grads[1]
        if choice_grad == nothing
            error("gradient not available for selected choice")
        end
        if retval_grad != nothing
            choice_grad += retval_grad
        end
        gradient_choices = ChoiceAtChoiceMap(trace.key, choice_grad)
    else
        value_choices = EmptyChoiceMap()
        gradient_choices = EmptyChoiceMap()
    end
    input_grads = (kernel_arg_grads[2:end]..., nothing)
    (input_grads, value_choices, gradient_choices)
end

function accumulate_param_gradients!(trace::ChoiceAtTrace, retval_grad)
    if retval_grad != nothing && !has_output_grad(trace.gen_fn.dist)
        error("return value gradient not accepted but one was provided")
    end
    kernel_arg_grads = logpdf_grad(trace.gen_fn.dist, trace.value, trace.kernel_args...)
    (kernel_arg_grads[2:end]..., nothing)
end

function to_serializable_trace(tr::ChoiceAtTrace)
    return GenericSerializableTrace(nothing, (tr.value, tr.key, tr.kernel_args, tr.score))
end
function from_serializable_trace(st::GenericSerializableTrace, gf::ChoiceAtCombinator)
    return get_trace_type(gf)(gf, st.properties...)
end

export choice_at
