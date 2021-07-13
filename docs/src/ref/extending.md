# Extending Gen

Gen is designed for extensibility.
To implement behaviors that are not directly supported by the existing modeling languages, users can implement `black-box' generative functions directly, without using built-in modeling language.
These generative functions can then be invoked by generative functions defined using the built-in modeling language.
This includes several special cases:

- Extending Gen with custom gradient computations

- Extending Gen with custom incremental computation of return values

- Extending Gen with new modeling languages.

## Custom gradients

To add a custom gradient for a differentiable deterministic computation, define a concrete subtype of [`CustomGradientGF`](@ref) with the following methods:

- [`apply`](@ref)

- [`gradient`](@ref)

- [`has_argument_grads`](@ref)

For example:

```julia
struct MyPlus <: CustomGradientGF{Float64} end

Gen.apply(::MyPlus, args) = args[1] + args[2]
Gen.gradient(::MyPlus, args, retval, retgrad) = (retgrad, retgrad)
Gen.has_argument_grads(::MyPlus) = (true, true)
```

```@docs
CustomGradientGF
apply
gradient
```

## Custom incremental computation

Iterative inference techniques like Markov chain Monte Carlo involve repeatedly updating the execution traces of generative models.
In some cases, the output of a deterministic computation within the model can be incrementally computed during each of these updates, instead of being computed from scratch.

To add a custom incremental computation for a deterministic computation, define a concrete subtype of [`CustomUpdateGF`](@ref) with the following methods:

- [`apply_with_state`](@ref)

- [`update_with_state`](@ref)

- [`has_argument_grads`](@ref)

The second type parameter of `CustomUpdateGF` is the type of the state that may be used internally to facilitate incremental computation within `update_with_state`.

For example, we can implement a function for computing the sum of a vector that efficiently computes the new sum when a small fraction of the vector elements change:

```julia
struct MyState
    prev_arr::Vector{Float64}
    sum::Float64
end

struct MySum <: CustomUpdateGF{Float64,MyState} end

function Gen.apply_with_state(::MySum, args)
    arr = args[1]
    s = sum(arr)
    state = MyState(arr, s)
    (s, state)
end

function Gen.update_with_state(::MySum, state, args, argdiffs::Tuple{VectorDiff})
    arr = args[1]
    prev_sum = state.sum
    retval = prev_sum
    for i in keys(argdiffs[1].updated)
        retval += (arr[i] - state.prev_arr[i])
    end
    prev_length = length(state.prev_arr)
    new_length = length(arr)
    for i=prev_length+1:new_length
        retval += arr[i]
    end
    for i=new_length+1:prev_length
        retval -= arr[i]
    end
    state = MyState(arr, retval)
    (state, retval, UnknownChange())
end

Gen.num_args(::MySum) = 1
```

```@docs
CustomUpdateGF
apply_with_state
update_with_state
```


## [Custom distributions](@id custom_distributions)

Users can extend Gen with new probability distributions, which can then be used
to make random choices within generative functions. Simple transformations of
existing distributions can be created using the [`@dist` DSL](@ref dist_dsl).
For arbitrary distributions, including distributions that cannot be expressed
in the `@dist` DSL, users can define a custom distribution by implementing
Gen's Distribution interface directly, as defined below.

Probability distributions are singleton types whose supertype is `Distribution{T}`, where `T` indicates the data type of the random sample.

```julia
abstract type Distribution{T} end
```

A new Distribution type must implement the following methods:

- [`random`](@ref)

- [`logpdf`](@ref)

- [`has_output_grad`](@ref)

- [`logpdf_grad`](@ref)

- [`has_argument_grads`](@ref)


By convention, distributions have a global constant lower-case name for the singleton value.
For example:

```julia
struct Bernoulli <: Distribution{Bool} end
const bernoulli = Bernoulli()
```
Distribution values should also be callable, which is a syntactic sugar with the same behavior of calling `random`:

```julia
bernoulli(0.5) # identical to random(bernoulli, 0.5) and random(Bernoulli(), 0.5)
```

```@docs
random
logpdf
has_output_grad
logpdf_grad
```

## Custom generative functions

We recommend the following steps for implementing a new type of generative function, and also looking at the implementation for the [`DynamicDSLFunction`](@ref) type as an example.

##### Define a trace data type
```julia
struct MyTraceType <: Trace
    ..
end
```

##### Decide the return type for the generative function
Suppose our return type is `Vector{Float64}`.

##### Define a data type for your generative function
This should be a subtype of [`GenerativeFunction`](@ref), with the appropriate type parameters.
```julia
struct MyGenerativeFunction <: GenerativeFunction{Vector{Float64},MyTraceType}
..
end
```
Note that your generative function may not need to have any fields.
You can create a constructor for it, e.g.:
```
function MyGenerativeFunction(...)
..
end
```

##### Decide what the arguments to a generative function should be
For example, our generative functions might take two arguments, `a` (of type `Int`) and `b` (of type `Float64`).
Then, the argument tuple passed to e.g. [`generate`](@ref) will have two elements.

NOTE: Be careful to distinguish between arguments to the generative function itself, and arguments to the constructor of the generative function.
For example, if you have a generative function type that is parametrized by, for example, modeling DSL code, this DSL code would be a parameter of the generative function constructor.

##### Decide what the traced random choices (if any) will be
Remember that each random choice is assigned a unique address in (possibly) hierarchical address space.
You are free to design this address space as you wish, although you should document it for users of your generative function type.

##### Implement methods of the Generative Function Interface

At minimum, you need to implement the following methods:

- [`simulate`](@ref)

- [`has_argument_grads`](@ref)

- [`accepts_output_grad`](@ref)

- [`get_args`](@ref)

- [`get_retval`](@ref)

- [`get_choices`](@ref)

- [`get_score`](@ref)

- [`get_gen_fn`](@ref)

- [`project`](@ref)

If you want to use the generative function within models, you should implement:

- [`generate`](@ref)

If you want to use MCMC on models that call your generative function, then implement:

- [`update`](@ref)

- [`regenerate`](@ref)

If you want to use gradient-based inference techniques on models that call your generative function, then implement:

- [`choice_gradients`](@ref)

- [`update`](@ref)

If your generative function has trainable parameters, then implement:

- [`accumulate_param_gradients!`](@ref)

#### Supporting trace serialization
To support trace serialization, a trace type of type `T` for a generative function of type `G` must convertable into a `SerializableTrace` object, and must be recoverable from a `SerializableTrace` object and the generative function.
```@docs
SerializableTrace
to_serializable_trace
from_serializable_trace
```
A user must implement `to_serializable_Trace(::T)`, and `from_serializable_Trace(::ST, ::G)` for some concrete type `ST <: SerializableTrace`.  This may be a custom type, or the user may use the built-in type
```@docs
GenericSerializableTrace
```

## Custom modeling languages

Gen can be extended with new modeling languages by implementing new generative function types, and constructors for these types that take models as input.
This typically requires implementing the entire generative function interface, and is advanced usage of Gen.
