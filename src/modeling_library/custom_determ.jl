##################
# CustomDetermGF #
##################

"""
    CustomDetermGFTrace{T,S} <: Trace

Trace type for custom deterministic generative function.
"""
struct CustomDetermGFTrace{T,S} <: Trace
    retval::T
    state::S
    args::Tuple
    gen_fn::Any
end

get_args(trace::CustomDetermGFTrace) = trace.args

get_retval(trace::CustomDetermGFTrace) = trace.retval

get_choices(trace::CustomDetermGFTrace) = EmptyChoiceMap()

get_score(trace::CustomDetermGFTrace) = 0.

project(trace::CustomDetermGFTrace, selection::Selection) = 0.

get_gen_fn(trace::CustomDetermGFTrace) = trace.gen_fn


"""
    CustomDetermGF{T,S} <: GenerativeFunction{T,CustomDetermGFTrace{T,S}}

Abstract type for a custom deterministic generative function.
"""
abstract type CustomDetermGF{T,S} <: GenerativeFunction{T,CustomDetermGFTrace{T,S}} end

# default implementation, can be overridden
accepts_output_grad(::CustomDetermGF) = false

"""
    retval, state = apply_with_state(gen_fn::CustomDetermGF, args)

Execute the generative function and return the return value and the state.
"""
function apply_with_state end

"""
    state, retval, retdiff = update_with_state(gen_fn::CustomDetermGF, state, args, argdiffs)

Update the arguments to the generative function and return new return value and state.
"""
function update_with_state(gen_fn::CustomDetermGF{T,S}, state, args, argdiffs) where {T,S}
    # default implementation, can be overridden
    new_retval, new_state = apply_with_state(gen_fn, args)
    retdiff = UnknownChange()
    (new_state, new_retval, retdiff)
end

"""
    arg_grads = gradient_with_state(gen_fn::CustomDetermGF, state, args, retgrad)

Return the gradient tuple with respect to the arguments.
"""
function gradient_with_state(gen_fn::CustomDetermGF, state, args, retgrad)
    # default implementation, can be overridden
    map((_) -> nothing, args)
end

"""
    arg_grads = accumulate_param_gradients_determ!(
        gen_fn::CustomDetermGF, state, args, retgrad, scale_factor)

Increment gradient accumulators for parameters the gradient with respect to the
arguments, optionally scaled, and return the gradient with respect to the
arguments (not scaled).

Given the previous state and a gradient with respect to the return value \$∇_y
J\$ (`retgrad`), return the following gradient (`arg_grads`) with respect to
the arguments \$x\$:
```math
∇_x J
```
Also increment the gradient accumulators for the trainable parameters \$Θ\$ of
the function by:
```math
s * ∇_Θ J
```
where \$s\$ is `scale_factor`.
"""
function accumulate_param_gradients_determ!(
        gen_fn::CustomDetermGF, state, args, retgrad, scale_factor)
    # default implementation, can be overridden
    gradient_with_state(gen_fn, state, args, retgrad)
end

function simulate(gen_fn::CustomDetermGF{T,S}, args::Tuple) where {T,S}
    retval, state = apply_with_state(gen_fn, args)
    CustomDetermGFTrace{T,S}(retval, state, args, gen_fn)
end

function generate(gen_fn::CustomDetermGF{T,S}, args::Tuple, choices::ChoiceMap) where {T,S}
    if !isempty(choices)
        error("Deterministic generative function makes no random choices")
    end
    retval, state = apply_with_state(gen_fn, args)
    trace = CustomDetermGFTrace{T,S}(retval, state, args, gen_fn)
    trace, 0.
end

function update(trace::CustomDetermGFTrace{T,S}, args::Tuple, argdiffs::Tuple, choices::ChoiceMap) where {T,S}
    if !isempty(choices)
        error("Deterministic generative function makes no random choices")
    end
    state, retval, retdiff = update_with_state(trace.gen_fn, trace.state, args, argdiffs)
    new_trace = CustomDetermGFTrace{T,S}(retval, state, args, trace.gen_fn)
    (new_trace, 0., retdiff, choicemap())
end

function regenerate(trace::CustomDetermGFTrace, args::Tuple, argdiffs::Tuple, selection::Selection)
    update(trace, args, argdiffs, EmptyChoiceMap())
end

function choice_gradients(trace::CustomDetermGFTrace, selection::Selection, retgrad)
    arg_grads = gradient_with_state(trace.gen_fn, trace.state, trace.args, retgrad)
    (arg_grads, EmptyChoiceMap(), EmptyChoiceMap())
end

function accumulate_param_gradients!(trace::CustomDetermGFTrace, retgrad, scale_factor)
    accumulate_param_gradients_determ!(trace.gen_fn, trace.state, trace.args, retgrad, scale_factor)
end

export CustomDetermGF, CustomDetermGFTrace, apply_with_state, update_with_state, gradient_with_state, accumulate_param_gradients_determ!

####################
# CustomGradientGF #
####################

"""
    CustomGradientGF{T}

Abstract type for a generative function with a custom gradient computation, and default behaviors for all other generative function interface methods.

`T` is the type of the return value.
"""
abstract type CustomGradientGF{T} <: CustomDetermGF{T,T} end

accepts_output_grad(::CustomGradientGF) = true

has_argument_grads(::CustomGradientGF) = error("not implemented")

"""
    retval = apply(gen_fn::CustomGradientGF, args)

Apply the function to the arguments.
"""
function apply(gen_fn::CustomGradientGF, args)
    error("not implemented")
end

function apply_with_state(gen_fn::CustomGradientGF, args)
    retval = apply(gen_fn, args)
    (retval, retval)
end

"""
    arg_grads = gradient(gen_fn::CustomDetermGF, args, retval, retgrad)

Return the gradient tuple with respect to the arguments, where `nothing` is for argument(s) whose gradient is not available.
"""
function gradient(gen_fn::CustomGradientGF, args, retval, retgrad)
    error("not implemented")
end

function gradient_with_state(gen_fn::CustomGradientGF, state, args, retgrad)
    retval = state
    gradient(gen_fn, args, retval, retgrad)
end

export CustomGradientGF, apply, gradient

##################
# CustomUpdateGF #
##################

"""
    CustomUpdateGF{T,S}

Abstract type for a generative function with a custom update computation, and default behaviors for all other generative function interface methods.

`T` is the type of the return value and `S` is the type of state used internally for incremental computation.
"""
abstract type CustomUpdateGF{T,S} <: CustomDetermGF{T,S} end

accepts_output_grad(::CustomUpdateGF) = false

"""
    num_args(::CustomUpdateGF)

Returns the number of arguments.
"""
num_args(::CustomUpdateGF) = error("not implemented")

has_argument_grads(gen_fn::CustomUpdateGF) = tuple(fill(nothing, num_args(gen_fn))...)

apply_with_state(gen_fn::CustomUpdateGF, args) = error("not implemented")

function to_serializable_trace(tr::CustomDetermGFTrace)
    return GenericSerializableTrace(nothing, (tr.retval, tr.state, tr.args))
end
function from_serializable_trace(st::GenericSerializableTrace, gf::CustomDetermGF)
    return get_trace_type(gf)(st.properties..., gf)
end

export CustomUpdateGF, num_args
