##########
# Traces #
##########

"""
    Trace

Abstract type for a trace of a generative function.
"""
abstract type Trace end

"""
    get_args(trace)

Return the argument tuple for a given execution.

Example:
```julia
args::Tuple = get_args(trace)
```
"""
function get_args end

"""
    get_retval(trace)

Return the return value of the given execution.

Example for generative function with return type `T`:
```julia
retval::T = get_retval(trace)
```
"""
function get_retval end

"""
    get_choices(trace)

Return a value implementing the assignment interface

Note that the value of any non-addressed randomness is not externally accessible.

Example:

```julia
choices::ChoiceMap = get_choices(trace)
z_val = choices[:z]
```
"""
function get_choices end

"""
    get_score(trace)

Return:
```math
\\log \\frac{p(t, r; x)}{q(r; x, t)}
```

When there is no non-addressed randomness, this simplifies to the log probability \$\\log p(t; x)\$.
"""
function get_score end

"""
    logpdf(trace)

Synonym for [`get_score`](@ref).
"""
logpdf(trace::Trace) = get_score(trace)

"""
    gen_fn::GenerativeFunction = get_gen_fn(trace)

Return the generative function that produced the given trace.
"""
function get_gen_fn end

"""
    value = getindex(trace::Trace, addr)

Get the value of the random choice, or auxiliary state (e.g. return value of inner function call), at address `addr`.
"""
function Base.getindex(trace::Trace, addr)
    get_choices(trace)[addr]
end

"""
    retval = getindex(trace::Trace)
    retval = trace[]

Synonym for [`get_retval`](@ref).
"""
Base.getindex(trace::Trace) = get_retval(trace)

export get_args
export get_retval
export get_choices
export get_score
export get_gen_fn

######################
# GenerativeFunction #
######################

"""
    GenerativeFunction{T,U <: Trace}

Abstract type for a generative function with return value type T and trace type U.
"""
abstract type GenerativeFunction{T,U <: Trace} end

get_return_type(::GenerativeFunction{T,U}) where {T,U} = T
get_trace_type(::GenerativeFunction{T,U}) where {T,U} = U

"""
    bools::Tuple = has_argument_grads(gen_fn::Union{GenerativeFunction,Distribution})

Return a tuple of booleans indicating whether a gradient is available for each of its arguments.
"""
function has_argument_grads end

"""
    req::Bool = accepts_output_grad(gen_fn::GenerativeFunction)

Return a boolean indicating whether the return value is dependent on any of the *gradient source elements* for any trace.

The gradient source elements are:

- Any argument whose position is true in `has_argument_grads`

- Any trainable parameter

- Random choices made at a set of addresses that are selectable by `choice_gradients`.
"""
function accepts_output_grad end

"""
    get_params(gen_fn::GenerativeFunction)

Return an iterable over the trainable parameters of the generative function.
"""
get_params(::GenerativeFunction) = ()

"""
    trace = simulate(gen_fn, args)

Execute the generative function and return the trace.

Given arguments (`args`), sample \$t \\sim p(\\cdot; x)\$ and \$r \\sim p(\\cdot; x,
t)\$, and return a trace with choice map \$t\$.

If `gen_fn` has optional trailing arguments (i.e., default values are provided),
the optional arguments can be omitted from the `args` tuple. The generated trace
 will have default values filled in.
"""
function simulate(::GenerativeFunction, ::Tuple)
    error("Not implemented")
end

"""
    (trace::U, weight) = generate(gen_fn::GenerativeFunction{T,U}, args::Tuple)

Return a trace of a generative function.

    (trace::U, weight) = generate(gen_fn::GenerativeFunction{T,U}, args::Tuple,
                                    constraints::ChoiceMap)

Return a trace of a generative function that is consistent with the given
constraints on the random choices.

Given arguments \$x\$ (`args`) and assignment \$u\$ (`constraints`) (which is empty for the first form), sample \$t \\sim
q(\\cdot; u, x)\$ and \$r \\sim q(\\cdot; x, t)\$, and return the trace \$(x, t, r)\$ (`trace`).
Also return the weight (`weight`):
```math
\\log \\frac{p(t, r; x)}{q(t; u, x) q(r; x, t)}
```

If `gen_fn` has optional trailing arguments (i.e., default values are provided),
the optional arguments can be omitted from the `args` tuple. The generated trace
 will have default values filled in.

Example without constraints:
```julia
(trace, weight) = generate(foo, (2, 4))
```

Example with constraint that address `:z` takes value `true`.
```julia
(trace, weight) = generate(foo, (2, 4), choicemap((:z, true))
```
"""
function generate(::GenerativeFunction, ::Tuple, ::ChoiceMap)
    error("Not implemented")
end

function generate(gen_fn::GenerativeFunction, args::Tuple)
    generate(gen_fn, args, EmptyChoiceMap())
end

"""
    weight = project(trace::U, selection::Selection)

Estimate the probability that the selected choices take the values they do in a
trace.

Given a trace \$(x, t, r)\$ (`trace`) and a set of addresses \$A\$ (`selection`),
let \$u\$ denote the restriction of \$t\$ to \$A\$. Return the weight
(`weight`):
```math
\\log \\frac{p(r, t; x)}{q(t; u, x) q(r; x, t)}
```
"""
function project(trace, selection::Selection)
    error("Not implemented")
end

"""
    (choices, weight, retval) = propose(gen_fn::GenerativeFunction, args::Tuple)

Sample an assignment and compute the probability of proposing that assignment.

Given arguments (`args`), sample \$t \\sim p(\\cdot; x)\$ and \$r \\sim p(\\cdot; x,
t)\$, and return \$t\$
(`choices`) and the weight (`weight`):
```math
\\log \\frac{p(r, t; x)}{q(r; x, t)}
```
"""
function propose(gen_fn::GenerativeFunction, args::Tuple)
    trace = simulate(gen_fn, args)
    weight = get_score(trace)
    (get_choices(trace), weight, get_retval(trace))
end

"""
    (weight, retval) = assess(gen_fn::GenerativeFunction, args::Tuple, choices::ChoiceMap)

Return the probability of proposing an assignment

Given arguments \$x\$ (`args`) and an assignment \$t\$ (`choices`) such that
\$p(t; x) > 0\$, sample \$r \\sim q(\\cdot; x, t)\$ and
return the weight (`weight`):
```math
\\log \\frac{p(r, t; x)}{q(r; x, t)}
```
It is an error if \$p(t; x) = 0\$.
"""
function assess(gen_fn::GenerativeFunction, args::Tuple, choices::ChoiceMap)
    (trace, weight) = generate(gen_fn, args, choices)
    (weight, get_retval(trace))
end

"""
    (new_trace, weight, retdiff, discard) = update(trace, args::Tuple, argdiffs::Tuple,
                                                   constraints::ChoiceMap)

Update a trace by changing the arguments and/or providing new values for some
existing random choice(s) and values for some newly introduced random choice(s).

Given a previous trace \$(x, t, r)\$ (`trace`), new arguments \$x'\$ (`args`), and
a map \$u\$ (`constraints`), return a new trace \$(x', t', r')\$ (`new_trace`)
that is consistent with \$u\$.  The values of choices in \$t'\$ are
either copied from \$t\$ or from \$u\$ (with \$u\$ taking
precedence) or are sampled from the internal proposal distribution.  All choices in \$u\$ must appear in \$t'\$.  Also return an
assignment \$v\$ (`discard`) containing the choices in \$t\$ that were
overwritten by values from \$u\$, and any choices in \$t\$ whose address does
not appear in \$t'\$. Sample \$t' \\sim q(\\cdot; x', t + u)\$, and \$r' \\sim
q(\\cdot; x', t')\$, where \$t + u\$ is the choice map obtained by merging
\$t\$ and \$u\$ with \$u\$ taking precedence for overlapping addresses.  Also
return a weight (`weight`):
```math
\\log \\frac{p(r', t'; x') q(r; x, t)}{p(r, t; x) q(r'; x', t') q(t'; x', t + u)}
```

Note that `argdiffs` is expected to be the same length as `args`. If the
function that generated `trace` supports default values for trailing arguments,
then these arguments can be omitted from `args` and `argdiffs`. Note
that if the original `trace` was generated using non-default argument values,
then for each optional argument that is omitted, the old value will be
over-written by the default argument value in the updated trace.
"""
function update(trace, args::Tuple, argdiffs::Tuple, ::ChoiceMap)
    error("Not implemented")
end

"""
    (new_trace, weight, retdiff, discard) = update(trace, constraints::ChoiceMap)

Shorthand variant of
[`update`](@ref update(::Any, ::Tuple, ::Tuple, ::ChoiceMap))
which assumes the arguments are unchanged.
"""
function update(trace, constraints::ChoiceMap)
    args = get_args(trace)
    argdiffs = Tuple(NoChange() for _ in args)
    return update(trace, args, argdiffs, constraints)
end

"""
    (new_trace, weight, retdiff) = regenerate(trace, args::Tuple, argdiffs::Tuple,
                                              selection::Selection)

Update a trace by changing the arguments and/or randomly sampling new values
for selected random choices using the internal proposal distribution family.

Given a previous trace \$(x, t, r)\$ (`trace`), new arguments \$x'\$ (`args`), and
a set of addresses \$A\$ (`selection`), return a new trace \$(x', t')\$
(`new_trace`) such that \$t'\$ agrees with \$t\$ on all addresses not in \$A\$
(\$t\$ and \$t'\$ may have different sets of addresses).  Let \$u\$ denote the
restriction of \$t\$ to the complement of \$A\$.  Sample \$t' \\sim Q(\\cdot;
u, x')\$ and sample \$r' \\sim Q(\\cdot; x', t')\$.
Return the new trace \$(x', t', r')\$ (`new_trace`) and the weight
(`weight`):
```math
\\log \\frac{p(r', t'; x') q(t; u', x) q(r; x, t)}{p(r, t; x) q(t'; u, x') q(r'; x', t')}
```
where \$u'\$ is the restriction of \$t'\$ to the complement of \$A\$.

Note that `argdiffs` is expected to be the same length as `args`. If the
function that generated `trace` supports default values for trailing arguments,
then these arguments can be omitted from `args` and `argdiffs`. Note
that if the original `trace` was generated using non-default argument values,
then for each optional argument that is omitted, the old value will be
over-written by the default argument value in the regenerated trace.
"""
function regenerate(trace, args::Tuple, argdiffs::Tuple, selection::Selection)
    error("Not implemented")
end

"""
    (new_trace, weight, retdiff) = regenerate(trace, selection::Selection)

Shorthand variant of
[`regenerate`](@ref regenerate(::Any, ::Tuple, ::Tuple, ::Selection))
which assumes the arguments are unchanged.
"""
function regenerate(trace, selection::Selection)
    args = get_args(trace)
    argdiffs = Tuple(NoChange() for _ in args)
    return regenerate(trace, args, argdiffs, selection)
end

"""
    arg_grads = accumulate_param_gradients!(trace, retgrad=nothing, scale_factor=1.)

Increment gradient accumulators for parameters by the gradient of the
log-probability of the trace, optionally scaled, and return the gradient with
respect to the arguments (not scaled).

Given a previous trace \$(x, t)\$ (`trace`) and a gradient with respect to the
return value \$∇_y J\$ (`retgrad`), return the following gradient (`arg_grads`)
with respect to the arguments \$x\$:
```math
∇_x \\left( \\log P(t; x) + J \\right)
```

The length of `arg_grads` will be equal to the number of arguments to the
function that generated `trace` (including any optional trailing arguments).
If an argument is not annotated with `(grad)`, the corresponding value in
`arg_grads` will be `nothing`.

Also increment the gradient accumulators for the trainable parameters \$Θ\$ of
the function by:
```math
∇_Θ \\left( \\log P(t; x) + J \\right)
```
"""
function accumulate_param_gradients!(trace, retgrad, scale_factor)
    error("Not implemented")
end

function accumulate_param_gradients!(trace, retgrad)
    accumulate_param_gradients!(trace, retgrad, 1.)
end

function accumulate_param_gradients!(trace)
    accumulate_param_gradients!(trace, nothing, 1.)
end

"""
    (arg_grads, choice_values, choice_grads) = choice_gradients(
        trace, selection=EmptySelection(), retgrad=nothing)

Given a previous trace \$(x, t)\$ (`trace`) and a gradient with respect to the
return value \$∇_y J\$ (`retgrad`), return the following gradient (`arg_grads`)
with respect to the arguments \$x\$:
```math
∇_x \\left( \\log P(t; x) + J \\right)
```

The length of `arg_grads` will be equal to the number of arguments to the
function that generated `trace` (including any optional trailing arguments).
If an argument is not annotated with `(grad)`, the corresponding value in
`arg_grads` will be `nothing`.

Also given a set of addresses \$A\$ (`selection`) that are continuous-valued
random choices, return the folowing gradient (`choice_grads`) with respect to
the values of these choices:
```math
∇_A \\left( \\log P(t; x) + J \\right)
```
The gradient is represented as a choicemap whose value at (hierarchical)
address `addr` is \$∂J/∂t[\\texttt{addr}]\$.

Also return the choicemap (`choice_values`) that is the restriction of \$t\$ to \$A\$.
"""
function choice_gradients(trace, selection::Selection, retgrad)
    error("Not implemented")
end

function choice_gradients(trace, selection::Selection)
    choice_gradients(trace, selection, nothing)
end

function choice_gradients(trace)
    choice_gradients(trace, EmptySelection(), nothing)
end

export GenerativeFunction
export Trace
export has_argument_grads
export accepts_output_grad
export get_params
export simulate
export generate
export project
export propose
export assess
export update
export regenerate
export accumulate_param_gradients!
export choice_gradients
export logpdf
