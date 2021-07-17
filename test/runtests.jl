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
        ans = (f(pos_args...) - f(neg_args...)) ./ (2. * dx)
        # Workaround for
        # https://github.com/probcomp/Gen.jl/pull/433#discussion_r669958584
        if args[i] isa AbstractArray && ndims(args[i]) == 0
            return fill(ans)
        end
        return ans
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

"""
Returns the partial derivatives of `f` with respect to all entries of
`args[i]`.

That is, returns an array of the same shape as `args[i]`, each entry of which
is [`finite_diff_arr`](@ref) applied to the corresponding entry of `args[i]`.

Requires that `args[i]` have nonzero rank.  Due to [1], handling
zero-dimensional arrays properly in this function is not feasible; the caller
should handle that case on their own.

[1] https://github.com/JuliaLang/julia/issues/28866
"""
function finite_diff_arr_fullarg(f::Function, args::Tuple, i::Int, dx::Float64)
    @assert args[i] isa AbstractArray
    @assert ndims(args[i]) > 0
    return [finite_diff_arr(f, args, i, idx, dx)
            for idx in keys(args[i])]
end

const dx = 1e-6

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
