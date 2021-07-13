##################
# map combinator #
##################

# used for type dispatch on the VectorTrace type (e.g. we will also have a UnfoldType)
struct MapType end

"""
    gen_fn = Map(kernel::GenerativeFunction)

Return a new generative function that applies the kernel independently for a vector of inputs.

The returned generative function has one argument with type `Vector{X}` for each argument of the input generative function with type `X`.
The length of each argument, which must be the same for each argument, determines the number of times the input generative function is called (N).
Each call to the input function is made under address namespace i for i=1..N.
The return value of the returned function has type `FunctionalCollections.PersistentVector{Y}` where `Y` is the type of the return value of the input function.
The map combinator is similar to the 'map' higher order function in functional programming, except that the map combinator returns a new generative function that must then be separately applied.

If `kernel` has optional trailing arguments, the corresponding `Vector` arguments can be omitted from calls to `Map(kernel)`.
"""
struct Map{T,U} <: GenerativeFunction{PersistentVector{T},VectorTrace{MapType,T,U}}
    kernel::GenerativeFunction{T,U}
end

export Map

has_argument_grads(map_gf::Map) = has_argument_grads(map_gf.kernel)
accepts_output_grad(map_gf::Map) = accepts_output_grad(map_gf.kernel)

function (gen_fn::Map)(args...)
    (_, _, retval) = propose(gen_fn, args)
    retval
end

function get_args_for_key(args::Tuple, key::Int)
    map((arg) -> arg[key], args)
end

function get_prev_and_new_lengths(args::Tuple, prev_trace)
    new_length = length(args[1])
    prev_args = get_args(prev_trace)
    prev_length = length(prev_args[1])
    (new_length, prev_length)
end

_gen_fn_at_addr(gf::Map, _) = gf.kernel

include("assess.jl")
include("propose.jl")
include("simulate.jl")
include("generate.jl")
include("generic_update.jl")
include("update.jl")
include("regenerate.jl")
include("backprop.jl")
