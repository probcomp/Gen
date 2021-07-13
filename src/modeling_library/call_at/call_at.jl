struct CallAtChoiceMap{K,T} <: ChoiceMap
    key::K
    submap::T
end

Base.isempty(choices::CallAtChoiceMap) = isempty(choices.submap)

function get_address_schema(::Type{T}) where {T<:CallAtChoiceMap}
    SingleDynamicKeyAddressSchema()
end

function get_submap(choices::CallAtChoiceMap{K,T}, addr::K) where {K,T}
    choices.key == addr ? choices.submap : EmptyChoiceMap()
end

get_submap(choices::CallAtChoiceMap, addr::Pair) = _get_submap(choices, addr)
get_value(choices::CallAtChoiceMap, addr::Pair) = _get_value(choices, addr)
has_value(choices::CallAtChoiceMap, addr::Pair) = _has_value(choices, addr)
get_submaps_shallow(choices::CallAtChoiceMap) = ((choices.key, choices.submap),)
get_values_shallow(::CallAtChoiceMap) = ()

# TODO optimize CallAtTrace using type parameters

struct CallAtTrace <: Trace
    gen_fn::GenerativeFunction # the ChoiceAtCombinator (not the kernel)
    subtrace::Any
    key::Any
end

function Base.getindex(trace::CallAtTrace, addr::Pair)
    first, rest = addr
    if first == trace.key
        return trace.subtrace[rest]
    else
        error("Address prefix $addr not found.")
    end
end
function Base.getindex(trace::CallAtTrace, addr)
    if addr == trace.key
        return trace.subtrace[]
    else
        error("Address $addr not found.")
    end
end

get_args(trace::CallAtTrace) = (get_args(trace.subtrace)..., trace.key)
get_retval(trace::CallAtTrace) = get_retval(trace.subtrace)
get_score(trace::CallAtTrace) = get_score(trace.subtrace)
get_gen_fn(trace::CallAtTrace) = trace.gen_fn

function get_choices(trace::CallAtTrace)
    CallAtChoiceMap(trace.key, get_choices(trace.subtrace))
end

struct CallAtCombinator{T,U,K} <: GenerativeFunction{T, CallAtTrace}
    kernel::GenerativeFunction{T,U}
end

function call_at(kernel::GenerativeFunction{T,U}, ::Type{K}) where {T,U,K}
    CallAtCombinator{T,U,K}(kernel)
end

function accepts_output_grad(gen_fn::CallAtCombinator)
    accepts_output_grad(gen_fn.kernel)
end

unpack_call_at_args(args) = (args[end], args[1:end-1])


function assess(gen_fn::CallAtCombinator, args::Tuple, choices::ChoiceMap)
    (key, kernel_args) = unpack_call_at_args(args)
    if length(collect(get_submaps_shallow(choices))) > 1 || !isempty(get_values_shallow(choices))
        error("Not all constraints were consumed")
    end
    submap = get_submap(choices, key)
    assess(gen_fn.kernel, kernel_args, submap)
end

function propose(gen_fn::CallAtCombinator, args::Tuple)
    (key, kernel_args) = unpack_call_at_args(args)
    (submap, weight, retval) = propose(gen_fn.kernel, kernel_args)
    choices = CallAtChoiceMap(key, submap)
    (choices, weight, retval)
end

function simulate(gen_fn::CallAtCombinator, args::Tuple)
    (key, kernel_args) = unpack_call_at_args(args)
    subtrace = simulate(gen_fn.kernel, kernel_args)
    CallAtTrace(gen_fn, subtrace, key)
end

function generate(gen_fn::CallAtCombinator{T,U,K}, args::Tuple,
                  choices::ChoiceMap) where {T,U,K}
    (key, kernel_args) = unpack_call_at_args(args)
    submap = get_submap(choices, key)
    (subtrace, weight) = generate(gen_fn.kernel, kernel_args, submap)
    trace = CallAtTrace(gen_fn, subtrace, key)
    (trace, weight)
end

function project(trace::CallAtTrace, selection::Selection)
    subselection = selection[trace.key]
    project(trace.subtrace, subselection)
end

function update(trace::CallAtTrace, args::Tuple, argdiffs::Tuple,
                choices::ChoiceMap)
    (key, kernel_args) = unpack_call_at_args(args)
    key_changed = (key != trace.key)
    submap = get_submap(choices, key)
    if key_changed
        (subtrace, weight) = generate(trace.gen_fn.kernel, kernel_args, submap)
        weight -= get_score(trace.subtrace)
        discard = get_choices(trace)
        retdiff = UnknownChange()
    else
        (subtrace, weight, retdiff, subdiscard) = update(
            trace.subtrace, kernel_args, argdiffs[1:end-1], submap)
        discard = CallAtChoiceMap(key, subdiscard)
    end
    new_trace = CallAtTrace(trace.gen_fn, subtrace, key)
    (new_trace, weight, retdiff, discard)
end

function regenerate(trace::CallAtTrace, args::Tuple, argdiffs::Tuple,
                    selection::Selection)
    (key, kernel_args) = unpack_call_at_args(args)
    key_changed = (key != trace.key)
    subselection = selection[key]
    if key_changed
        if !isempty(subselection)
            error("Cannot select addresses under new key $key in regenerate")
        end
        (subtrace, weight) = generate(trace.gen_fn.kernel, kernel_args, EmptyChoiceMap())
        weight -= project(trace.subtrace, EmptySelection())
        retdiff = UnknownChange()
    else
        (subtrace, weight, retdiff) = regenerate(
            trace.subtrace, kernel_args, argdiffs[1:end-1], subselection)
    end
    new_trace = CallAtTrace(trace.gen_fn, subtrace, key)
    (new_trace, weight, retdiff)
end

function choice_gradients(trace::CallAtTrace, selection::Selection, retval_grad)
    subselection = selection[trace.key]
    (kernel_input_grads, value_submap, gradient_submap) = choice_gradients(
        trace.subtrace, subselection, retval_grad)
    input_grads = (kernel_input_grads..., nothing)
    value_choices = CallAtChoiceMap(trace.key, value_submap)
    gradient_choices = CallAtChoiceMap(trace.key, gradient_submap)
    (input_grads, value_choices, gradient_choices)
end

function accumulate_param_gradients!(trace::CallAtTrace, retval_grad)
    kernel_input_grads = accumulate_param_gradients!(trace.subtrace, retval_grad)
    (kernel_input_grads..., nothing)
end

function to_serializable_trace(tr::CallAtTrace)
    return GenericSerializableTrace(to_serializable_trace(tr.subtrace), tr.key)
end
function from_serializable_trace(st::GenericSerializableTrace, gf::CallAtCombinator)
    return get_trace_type(gf)(gf, from_serializable_trace(st.subtraces, gf.kernel), st.properties)
end

export call_at
