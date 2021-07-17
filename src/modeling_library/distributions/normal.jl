struct Normal <: Distribution{Float64} end
struct BroadcastedNormal <: Distribution{Array{Float64}} end

function broadcast_shapes_or_crash(args...)
    # This expression produces a DimensionMismatch exception if the shapes are
    # incompatible.  There seems to currently be no ready-made exception-free
    # way to check this, and I think it makes more sense to wait until that
    # capability exists in Julia proper than to re-implement it here.
    s = Base.Broadcast.broadcast_shape(map(size, args)...)
end

function assert_has_shape(x, expected_shape; msg="Shape assertion failed")
    if size(x) != expected_shape
        throw(DimensionMismatch(string(msg,
                                       " Expected shape: $expected_shape",
                                       " Actual shape: $(size(x))")))
    end
    nothing
end


"""
    normal(mu::Real, std::Real)

Samples a `Float64` value from a normal distribution.
"""
const normal = Normal()

"""
    broadcasted_normal(mu::AbstractArray{<:Real, N1},
                       std::AbstractArray{<:Real, N2}) where {N1, N2}

Samples an `Array{Float64, max(N1, N2)}` of shape
`Broadcast.broadcast_shapes(size(mu), size(std))` where each element is
independently normally distributed.  This is equivalent to (a reshape of) a
multivariate normal with diagonal covariance matrix, but its implementation is
more efficient than that of the more general [`mvnormal`](@ref) for this case.

The shapes of `mu` and `std` must be broadcast-compatible.

If all args are 0-dimensional arrays, then sampling via
`broadcasted_normal(...)` returns a `Float64` rather than properly returning an
`Array{Float64, 0}`.  This is consistent with Julia's own inconsistency on the
matter:

```jldoctest
julia> typeof(ones())
Array{Float64,0}

julia> typeof(ones() .* ones())
Float64
```
"""
const broadcasted_normal = BroadcastedNormal()

function logpdf(::Normal, x::Real, mu::Real, std::Real)
    z = (x - mu) / std
    - (abs2(z) + log(2π))/2 - log(std)
end

function logpdf(::BroadcastedNormal,
                x::Union{AbstractArray{<:Real}, Real},
                mu::Union{AbstractArray{<:Real}, Real},
                std::Union{AbstractArray{<:Real}, Real})
    assert_has_shape(x, broadcast_shapes_or_crash(mu, std);
                     msg="Shape of `x` does not agree with the sample space")
    z = (x .- mu) ./ std
    sum(- (abs2.(z) .+ log(2π)) / 2 .- log.(std))
end

function logpdf_grad(::Normal, x::Real, mu::Real, std::Real)
    z = (x - mu) / std
    deriv_x = - z / std
    deriv_mu = -deriv_x
    deriv_std = -1. / std + abs2(z) / std
    (deriv_x, deriv_mu, deriv_std)
end

function logpdf_grad(::BroadcastedNormal,
                     x::Union{AbstractArray{<:Real}, Real},
                     mu::Union{AbstractArray{<:Real}, Real},
                     std::Union{AbstractArray{<:Real}, Real})
    assert_has_shape(x, broadcast_shapes_or_crash(mu, std);
                     msg="Shape of `x` does not agree with the sample space")
    z = (x .- mu) ./ std
    deriv_x = - z ./ std
    deriv_mu = -deriv_x
    deriv_std = -1. ./ std .+ abs2.(z) ./ std
    (_unbroadcast_like(x, deriv_x), 
     _unbroadcast_like(mu, deriv_mu), 
     _unbroadcast_like(std, deriv_std))
end

_unbroadcast_like(::Real, full_arr) = sum(full_arr)
_unbroadcast_like(::AbstractArray{<:Real, 0}, full_arr::Real) = fill(full_arr)
function _unbroadcast_like(a::AbstractArray{<:Real, N},
                           full_arr::AbstractArray{T}
                          )::AbstractArray{T, N} where {N,T}
    if size(a) == size(full_arr)
        return full_arr
    end
    return _unbroadcast_to_shape(size(a), full_arr)
end

"""
"Unbroadcasts" `full_arr` to have shape `target_shape` by:

* Summing over all dims that would be increased by a broadcast from shape
  `target_shape` to shape `size(full_arr)`
* Then dropping trailing dims (which will all be 1's) as needed so that the
  result has shape `target_shape`.

Requires that `size(full_arr)` is "strictly bigger" than `target_shape`, in the
sense that

    Broadcast.broadcast_shapes(target_shape, size(full_arr)) == size(full_arr)
"""
function _unbroadcast_to_shape(target_shape::NTuple{target_ndims, Int},
                               full_arr::AbstractArray{T, full_ndims}
                         ) where {T, target_ndims, full_ndims}
    @assert full_ndims >= target_ndims
    should_sum_dim(i) = (i > target_ndims) || (target_shape[i] == 1 &&
                                               size(full_arr, i) > 1)
    dropdims(sum(full_arr; dims=filter(should_sum_dim, 1:full_ndims));
             dims=Dims(target_ndims + 1 : full_ndims))
end

random(::Normal, mu::Real, std::Real) = mu + std * randn()
is_discrete(::Normal) = false

function random(::BroadcastedNormal,
                mu::Union{AbstractArray{<:Real}, Real},
                std::Union{AbstractArray{<:Real}, Real})
    broadcast_shape = broadcast_shapes_or_crash(mu, std)
    mu .+ std .* randn(broadcast_shape)
end

(::Normal)(mu, std) = random(Normal(), mu, std)
(::BroadcastedNormal)(mu, std) = random(BroadcastedNormal(), mu, std)

has_output_grad(::Normal) = true
has_argument_grads(::Normal) = (true, true)

has_output_grad(::BroadcastedNormal) = true
has_argument_grads(::BroadcastedNormal) = (true, true)

export normal
export broadcasted_normal
