struct Switch{C, N, K, T} <: GenerativeFunction{T, Trace}
    branches::NTuple{N, GenerativeFunction{T}}
    cases::Dict{C, Int}
    function Switch(gen_fns::GenerativeFunction...)
        @assert !isempty(gen_fns)
        rettype = get_return_type(getindex(gen_fns, 1))
        new{Int, length(gen_fns), typeof(gen_fns), rettype}(gen_fns, Dict{Int, Int}())
    end
    function Switch(d::Dict{C, Int}, gen_fns::GenerativeFunction...) where C
        @assert !isempty(gen_fns)
        rettype = get_return_type(getindex(gen_fns, 1))
        new{C, length(gen_fns), typeof(gen_fns), rettype}(gen_fns, d)
    end
end
export Switch

has_argument_grads(switch_fn::Switch) = map(zip(map(has_argument_grads, switch_fn.branches)...)) do as
    all(as)
end
accepts_output_grad(switch_fn::Switch) = all(accepts_output_grad, switch_fn.branches)

function (gen_fn::Switch)(index::Int, args...)
    (_, _, retval) = propose(gen_fn, (index, args...))
    retval
end

function (gen_fn::Switch{C})(index::C, args...) where C
    (_, _, retval) = propose(gen_fn, (gen_fn.cases[index], args...))
    retval
end

include("trace.jl")

function to_serializable_trace(tr::SwitchTrace)
    GenericSerializableTrace(to_serializable_trace(tr.branch), (tr.index, tr.retval, tr.args, tr.score, tr.noise))
end
function from_serializable_trace(c::GenericSerializableTrace, gf::Switch)
    (index, retval, args, score, noise) = c.properties
    GenericSerializableTrace(
        gf, index,
        from_serializable_trace(c.subtraces, gf.branches[index]),
        retval, args, score, noise
    )
end

include("assess.jl")
include("propose.jl")
include("simulate.jl")
include("generate.jl")
include("update.jl")
include("regenerate.jl")
include("backprop.jl")

@doc(
"""
    gen_fn = Switch(gen_fns::GenerativeFunction...)

Returns a new generative function that accepts an argument tuple of type `Tuple{Int, ...}` where the first index indicates which branch to call.

    gen_fn = Switch(d::Dict{T, Int}, gen_fns::GenerativeFunction...) where T

Returns a new generative function that accepts an argument tuple of type `Tuple{Int, ...}` or an argument tuple of type `Tuple{T, ...}` where the first index either indicates which branch to call, or indicates an index into `d` which maps to the selected branch. This form is meant for convenience - it allows the programmer to use `d` like if-else or case statements.

`Switch` is designed to allow for the expression of patterns of if-else control flow. `gen_fns` must satisfy a few requirements:

1. Each `gen_fn` in `gen_fns` must accept the same argument types.
2. Each `gen_fn` in `gen_fns` must return the same return type.

Otherwise, each `gen_fn` can come from different modeling languages, possess different traces, etc.
""", Switch)
