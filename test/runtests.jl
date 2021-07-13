using Gen
using Test
import Random

"""
Compute a numerical partial derivative of `f` with respect to the `i`th
argument using finite differences.

If `broadcast` is `false`, then `args[i]` must be a scalar.

If `broadcast` is `true`, then `args[i]` may be an array.  In that case, `f` is
assumed to be the broadcast of a function whose `i`th argument is a scalar.  In
particular, `f` must still operate independently on each element of `args[i]`.
This condition cannot be checked automatically for arbitrary `f::Function`, so
the caller must guarantee it.
"""
function finite_diff(f::Function, args::Tuple, i::Int, dx::Float64;
                     broadcast=false)
    pos_args = Any[args...]
    neg_args = Any[args...]
    if broadcast
        pos_args[i] = copy(args[i]) .+ dx
        neg_args[i] = copy(args[i]) .- dx
        return (f(pos_args...) - f(neg_args...)) ./ (2. * dx)
    else
        pos_args[i] += dx
        neg_args[i] -= dx
        return (f(pos_args...) - f(neg_args...)) / (2. * dx)
    end
end

function finite_diff_vec(f::Function, args::Tuple, i::Int, j::Int, dx::Float64)
    pos_args = Any[deepcopy(args)...]
    pos_args[i][j] += dx
    neg_args = Any[deepcopy(args)...]
    neg_args[i][j] -= dx
    return (f(pos_args...) - f(neg_args...)) / (2. * dx)
end

"""
Compute a numerical partial derivative of `f` with respect to the `i`th argument, a symettric matrix, using finite differences.

Finite differences are applied symettrically to off-diagonal elements of the matrix, ensuring that the modified matrix is still symettric.

When applied to off-diagonal elements, the finite difference is scaled by a factor of 1/2.

The condition that the `i`th argument is a symettric matrix cannot be checked automatically, so the caller must guarantee it.
"""

function finite_diff_mat_sym(f::Function, args::Tuple, i::Int, j::Int, k::Int, dx::Float64)
    pos_args = Any[deepcopy(args)...]
    pos_args[i][j, k] += dx
    neg_args = Any[deepcopy(args)...]
    neg_args[i][j, k] -= dx

    if j!=k
        pos_args[i][k, j] += dx
        neg_args[i][k, j] -= dx
        return (f(pos_args...) - f(neg_args...)) / (4. * dx)
    end

    return (f(pos_args...) - f(neg_args...)) / (2. * dx)
end

"""
Compute a numerical partial derivative of `f` with respect to `idx`th element in the `i`th
argument, which must be array-valued, using finite differences.
"""
function finite_diff_arr(f::Function, args::Tuple, i::Int, idx, dx::Float64)
    pos_args = Any[deepcopy(args)...]
    pos_args[i][idx] += dx
    neg_args = Any[deepcopy(args)...]
    neg_args[i][idx] -= dx
    return (f(pos_args...) - f(neg_args...)) / (2. * dx)
end

const dx = 1e-6

"""
Attempts to serialize then deserialize the given trace, and returns
whether the pre-serialization and post-serialization traces are equal.
"""
function serialize_loop_successful(tr)
    io = IOBuffer()
    serialize_trace(io, tr)
    seek(io, 0)
    des_tr = deserialize_trace(io, get_gen_fn(tr))

    if get_choices(des_tr) != get_choices(tr)
        display(tr)
        display(des_tr)
    end

    return get_choices(des_tr) == get_choices(tr)
end

include("autodiff.jl")
include("diff.jl")
include("selection.jl")
include("assignment.jl")
include("gen_fn_interface.jl")
include("dsl/dsl.jl")
include("optional_args.jl")
include("static_ir/static_ir.jl")
include("tilde_sugar.jl")
include("inference/inference.jl")
include("modeling_library/modeling_library.jl")